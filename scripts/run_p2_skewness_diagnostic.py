import numpy as np

class P2Quantile:
    """
    Minimal P² quantile estimator (5 markers), suitable for diagnostics.
    This is not meant to replace your repo implementation—use it only if you
    cannot easily call the existing P² estimator objects.
    """
    def __init__(self, phi: float):
        assert 0.0 < phi < 1.0
        self.phi = phi
        self.n = 0
        self.q = None
        self.ni = None
        self.ni_des = None

    def update(self, x: float):
        if self.n < 5:
            if self.q is None:
                self.q = []
            self.q.append(x)
            self.n += 1
            if self.n == 5:
                self.q.sort()
                self.q = np.array(self.q, dtype=np.float64)
                self.ni = np.array([1, 2, 3, 4, 5], dtype=np.int64)
                self.ni_des = self._desired_positions(5)
            return

        # Find cell k
        k = 0
        if x < self.q[0]:
            self.q[0] = x
            k = 0
        elif x >= self.q[4]:
            self.q[4] = x
            k = 3
        else:
            for i in range(1, 5):
                if self.q[i - 1] <= x < self.q[i]:
                    k = i - 1
                    break

        self.n += 1
        for i in range(k + 1, 5):
            self.ni[i] += 1

        self.ni_des = self._desired_positions(self.n)

        for i in [1, 2, 3]:
            d = self.ni_des[i] - self.ni[i]
            if (d >= 1 and self.ni[i + 1] - self.ni[i] > 1) or (d <= -1 and self.ni[i] - self.ni[i - 1] > 1):
                di = 1 if d >= 1 else -1
                q_new = self._parabolic(i, di)
                if self.q[i - 1] < q_new < self.q[i + 1]:
                    self.q[i] = q_new
                else:
                    self.q[i] = self._linear(i, di)
                self.ni[i] += di

    def value(self) -> float:
        if self.n == 0:
            return np.nan
        if self.n < 5:
            # crude fallback
            arr = np.sort(np.array(self.q, dtype=np.float64))
            idx = int(np.floor(self.phi * (len(arr) - 1)))
            return float(arr[idx])
        return float(self.q[2])

    def _desired_positions(self, N: int):
        phi = self.phi
        # Common P² desired positions for 5 markers:
        # n'_0 = 1
        # n'_1 = 1 + 2*phi*(N-1)
        # n'_2 = 1 + 4*phi*(N-1)/2 = 1 + 2*phi*(N-1)
        # n'_3 = 1 + 2*(1+phi)*(N-1)
        # n'_4 = N
        # (We implement a consistent monotone scheme.)
        return np.array([
            1,
            1 + 2 * phi * (N - 1),
            1 + 4 * phi * (N - 1) / 2.0,
            1 + 2 * (1 + phi) * (N - 1),
            N
        ], dtype=np.float64)

    def _parabolic(self, i: int, d: int) -> float:
        q, n = self.q, self.ni.astype(np.float64)
        num = d / (n[i + 1] - n[i - 1])
        a = (n[i] - n[i - 1] + d) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
        b = (n[i + 1] - n[i] - d) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
        return q[i] + num * (a + b)

    def _linear(self, i: int, d: int) -> float:
        q, n = self.q, self.ni.astype(np.float64)
        return q[i] + d * (q[i + d] - q[i]) / (n[i + d] - n[i])


def run_skew_diagnostic(dist_name: str, sampler, phis=(0.01, 0.25, 0.5, 0.75, 0.99), Ns=(200, 500, 1000, 2000, 5000), reps=30):
    print(f"\n== {dist_name} ==")
    for N in Ns:
        errs = {phi: [] for phi in phis}
        for _ in range(reps):
            x = sampler(N)
            # exact quantiles (reference)
            q_ref = {phi: float(np.quantile(x, phi)) for phi in phis}
            # P2 estimates
            ests = {phi: P2Quantile(phi) for phi in phis}
            for val in x:
                for phi in phis:
                    ests[phi].update(float(val))
            for phi in phis:
                errs[phi].append(abs(ests[phi].value() - q_ref[phi]))
        msg = "N={:5d} | ".format(N) + " ".join([f"phi={phi:0.2f}: MAE={np.mean(errs[phi]):.4g}" for phi in phis])
        print(msg)


def main():
    rng = np.random.default_rng(0)

    run_skew_diagnostic(
        "Lognormal(0,1) (right-skew)",
        lambda N: rng.lognormal(mean=0.0, sigma=1.0, size=N),
    )
    run_skew_diagnostic(
        "Skew-normal(a=10) approx (right-skew via exp)",
        lambda N: np.exp(rng.normal(size=N)),
    )
    run_skew_diagnostic(
        "Left-skew (negated lognormal)",
        lambda N: -rng.lognormal(mean=0.0, sigma=1.0, size=N),
    )


if __name__ == "__main__":
    main()
