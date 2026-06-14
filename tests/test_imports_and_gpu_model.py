import os
import subprocess
import sys
import textwrap
import threading

from func2stream import gpu_model


def test_basic_package_import_does_not_require_opencv():
    code = r'''
import builtins

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cv2":
        raise ModuleNotFoundError("No module named 'cv2'", name="cv2")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

import func2stream

assert func2stream.Pipeline.__name__ == "Pipeline"
assert func2stream.DataSource.__name__ == "DataSource"

try:
    func2stream.VideoSource("0")
except ModuleNotFoundError as exc:
    assert "func2stream[video]" in str(exc)
else:
    raise AssertionError("VideoSource should report the optional OpenCV dependency")
'''

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        env=env,
    )


def test_gpu_model_creates_one_model_per_thread():
    creations = []

    class ProbeModel:
        def __init__(self):
            self.created_in = threading.get_ident()
            creations.append(self.created_in)

        def __call__(self, value):
            return value, self.created_in

    proxy = gpu_model(lambda: ProbeModel())

    main_value, main_thread = proxy("main")
    worker_results = []

    def run_worker():
        worker_results.append(proxy("worker"))

    worker = threading.Thread(target=run_worker)
    worker.start()
    worker.join()

    assert main_value == "main"
    assert worker_results[0][0] == "worker"
    assert main_thread != worker_results[0][1]
    assert creations == [main_thread, worker_results[0][1]]
