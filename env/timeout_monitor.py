import faulthandler
import os
import threading
import time

import gym
from PIL import Image


class TimeoutMonitor(gym.Wrapper):
    """ Timeout Monitor for Environment lockup bug tracking
    This class was created to work around/debug issues with the environment locking up while training
    or evaluating.
    """
    def __init__(self, env, memory):
        gym.Wrapper.__init__(self, env)
        self._memory = memory
        self._running = True
        self._waiting = False
        self._killed = False
        self._timeout = 30
        self._output_dir = './'
        self._cv = threading.Condition()
        self._thread = threading.Thread(target=self._run)
        self._thread.setDaemon(True)
        self._thread.start()

    def step(self, action):
        self._set_waiting()
        result = self.env.step(action)
        self._clear_waiting()
        return result

    def reset(self, **kwargs):
        self._set_waiting()
        result = self.env.reset(**kwargs)
        if self._killed:
            self._killed = False
            time.sleep(45)
            raise TimeoutError()
        self._clear_waiting()
        return result

    def close(self):
        with self._cv:
            self._running = False
            self._waiting = False
            self._cv.notify_all()
        self._thread.join()
        return self.env.close()

    def _set_waiting(self):
        with self._cv:
            assert not self._waiting
            self._waiting = True
            self._cv.notify()

    def _clear_waiting(self):
        with self._cv:
            assert self._waiting
            self._waiting = False
            self._cv.notify()

    def _run(self):
        with self._cv:
            print('Timeout monitor active...')
            while self._running:
                self._cv.wait_for(lambda: self._waiting or not self._running)
                if not self._running:
                    break
                not_expired = self._cv.wait_for(lambda: not self._waiting or not self._running, self._timeout)
                if not not_expired:
                    print('TIMEOUT!')
                    self._dump_memory()
                    with open('./freeze-trace.txt', "w+") as f:
                        faulthandler.dump_traceback(f)
                    self._killed = True
                    self.unwrapped.proc1.kill()
                    time.sleep(15)

    def _dump_memory(self, n=60):
        mem_start_idx = self._memory.transitions.index
        mem_size = self._memory.transitions.size
        for i in range(n):
            idx = (mem_start_idx - i - 1) % mem_size
            t = self._memory.transitions.data[idx]
            if t is not None:
                print('Dumping frame %d at timestep %d, performing action %d.' % (idx, t.timestep, t.action))
                state_np = t.state.permute(1, 2, 0).numpy()
                Image.fromarray(state_np).save(os.path.join(self._output_dir, 'frame-t%d-i%d.png' % (idx, t.timestep)))
            else:
                print('Invalid memory at ', idx)
