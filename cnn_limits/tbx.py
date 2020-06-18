import time

class PrintTimings:
    @staticmethod
    def hhmmss(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h == 0.0:
            return f"{m:02d}:{s:02d}"
        else:
            return f"{h:02d}:{m:02d}:{s:02d}"

    def __init__(self, desc="time", print_interval=2.):
        self.desc = desc
        self.print_interval = print_interval
        self.data = []

    def __call__(self, iterator, total=None):
        """
        Prints the current total number of iterations, speed of iteration, and
        elapsed time.

        Meant as a rudimentary replacement for `tqdm` that prints a new line at
        each iteration, and thus can be used in multiple parallel processes in the
        same terminal.
        """
        start_time = time.perf_counter()
        if total is None:
            total = len(iterator)
        last_printed = -self.print_interval
        prev_i = -1
        for i, value in enumerate(iterator):
            yield value
            cur_time = time.perf_counter()
            elapsed = cur_time - start_time
            if elapsed > last_printed + self.print_interval:
                it_s = (i - prev_i)/(elapsed - last_printed)
                it_left = total - i
                total_s = elapsed + it_left/it_s

                print((f"{self.desc}: {i+1}/{total} it, {it_s:.02f} it/s, "
                       f"[{self.hhmmss(elapsed)}<{self.hhmmss(total_s)}] ")
                      + ", ".join(f"{k}={v}" for k, v in self.data))
                last_printed = elapsed
                prev_i = i
