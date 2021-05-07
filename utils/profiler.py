import cProfile
import io
import pstats
from typing import Callable, Optional


def profile(file_path: Optional[str] = None):
    def profile_decorator(fnc: Callable):
        def inner(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            retval = fnc(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats()
            if file_path:
                with open(file_path, "wt") as f:
                    print(s.getvalue(), file=f)
            else:
                print(s.getvalue())
            return retval

        return inner

    return profile_decorator
