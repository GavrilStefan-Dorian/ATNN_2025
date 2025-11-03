import os
import subprocess
import filelock

def build():
    p_dir = os.path.dirname(__file__)
    lock_path = os.path.join(p_dir, ".lock")

    try:
        import transf_in_cpp
        print("transf_in_cpp exists already")
        return
    except ImportError:
        pass

    with filelock.FileLock(lock_path):
        print("transf_in_cpp is being built")
        subprocess.run(["python", "setup.py", "build_ext", "--inplace"], cwd=p_dir, check=True)
        print("transf_in_cpp was built!")

if __name__ == "__main__":
    build()
