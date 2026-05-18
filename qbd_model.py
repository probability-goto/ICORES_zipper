import numpy as np

class DynamicThresholdQBD:
    def __init__(self, lam1, lam2, mu1_rand, mu2_rand, mu1_zip, mu2_zip, K, m, p):
        # パラメータの初期化
        self.lam1 = lam1
        self.lam2 = lam2
        self.mu1_rand = mu1_rand
        self.mu2_rand = mu2_rand
        self.mu1_zip = mu1_zip
        self.mu2_zip = mu2_zip
        self.K = K
        self.m = m
        self.p = p
        
        # フェーズ数の定義
        self.num_phases = 2 * K + 1      # レベル n1 >= 1 のフェーズ数
        self.num_phases_0 = K + 1        # レベル n1 == 0 のフェーズ数

        # 行列の生成メソッドを呼び出し
        self.Q_plus1, self.Q_minus1, self.Q0 = self._build_Q_matrices()
        self.B_plus1, self.B_minus1, self.B0 = self._build_B_matrices()

    def _idx(self, s, n2):
        """レベル1以上における状態 (s, n2) のインデックスを返す"""
        if n2 == 0:
            return 0  # (1, 0)
        return 2 * n2 - 1 if s == 1 else 2 * n2

    def _idx0(self, n2):
        """
        レベル0における状態のインデックスを返す。
        レベル0（本線が空）のとき、サーバーが本線を処理する状態(S=1)は存在しない。
        n2=0 のときはシステム全体が空(アイドル状態 S=0)となる。
        """
        if n2 == 0:
            return 0  # (0, 0)
        return n2     # (2, n2)

    def _build_Q_matrices(self):
        """レベル n1 >= 1 における推移速度行列 Q_plus1, Q_minus1, Q0 を構築"""
        size = self.num_phases
        Q_plus1 = np.zeros((size, size))
        Q_minus1 = np.zeros((size, size))
        Q0 = np.zeros((size, size))

        # --- Q_plus1 の構築 (到着によるレベル増加) ---
        np.fill_diagonal(Q_plus1, self.lam1)

        # --- Q_minus1 と Q0 の非対角成分の構築 ---
        for n2 in range(self.K + 1):
            # 1. 本線のサービス完了 (Q_minus1に記述)
            if n2 == 0:
                Q_minus1[self._idx(1, 0), self._idx(1, 0)] = self.mu1_rand
            elif 1 <= n2 <= self.m:
                Q_minus1[self._idx(1, n2), self._idx(1, n2)] = self.p * self.mu1_rand
                Q_minus1[self._idx(1, n2), self._idx(2, n2)] = (1 - self.p) * self.mu1_rand
            else: # m < n2 <= K
                Q_minus1[self._idx(1, n2), self._idx(2, n2)] = self.mu1_zip

            # 2. 合流車線への到着 (Q0に記述)
            if n2 < self.K:
                if n2 == 0:
                    Q0[self._idx(1, 0), self._idx(1, 1)] = self.lam2
                else:
                    Q0[self._idx(1, n2), self._idx(1, n2 + 1)] = self.lam2
                    Q0[self._idx(2, n2), self._idx(2, n2 + 1)] = self.lam2

            # 3. 合流車線のサービス完了 (Q0に記述)
            if n2 >= 1:
                if n2 == 1:
                    Q0[self._idx(2, 1), self._idx(1, 0)] = self.mu2_rand
                elif 2 <= n2 <= self.m:
                    Q0[self._idx(2, n2), self._idx(1, n2 - 1)] = self.p * self.mu2_rand
                    Q0[self._idx(2, n2), self._idx(2, n2 - 1)] = (1 - self.p) * self.mu2_rand
                else: # m < n2 <= K
                    Q0[self._idx(2, n2), self._idx(1, n2 - 1)] = self.mu2_zip

        # --- Q0 の対角成分の構築 (各行の和が0になるように設定) ---
        for i in range(size):
            out_rate = np.sum(Q0[i, :]) + np.sum(Q_plus1[i, :]) + np.sum(Q_minus1[i, :])
            Q0[i, i] = -out_rate

        return Q_plus1, Q_minus1, Q0

    def _build_B_matrices(self):
        """境界条件（レベル0）に関連する行列 B_plus1, B_minus1, B0 を構築"""
        size0 = self.num_phases_0
        size1 = self.num_phases
        B_plus1 = np.zeros((size0, size1))
        B_minus1 = np.zeros((size1, size0))
        B0 = np.zeros((size0, size0))

        # --- B_plus1 の構築 (レベル0 -> 1) ---
        B_plus1[self._idx0(0), self._idx(1, 0)] = self.lam1
        for n2 in range(1, self.K + 1):
            B_plus1[self._idx0(2, n2), self._idx(2, n2)] = self.lam1

        # --- B_minus1 の構築 (レベル1 -> 0) ---
        B_minus1[self._idx(1, 0), self._idx0(0, 0)] = self.mu1_rand
        for n2 in range(1, self.K + 1):
            if n2 <= self.m:
                B_minus1[self._idx(1, n2), self._idx0(2, n2)] = self.mu1_rand
            else:
                B_minus1[self._idx(1, n2), self._idx0(2, n2)] = self.mu1_zip

        # --- B0 の非対角成分の構築 (レベル0内での遷移) ---
        for n2 in range(self.K + 1):
            # 1. 合流車線への到着
            if n2 < self.K:
                if n2 == 0:
                    B0[self._idx0(0, 0), self._idx0(2, 1)] = self.lam2
                else:
                    B0[self._idx0(2, n2), self._idx0(2, n2 + 1)] = self.lam2
            
            # 2. 合流車線のサービス完了
            if n2 >= 1:
                if n2 <= self.m:
                    B0[self._idx0(2, n2), self._idx0(2, n2 - 1) if n2 > 1 else self._idx0(0, 0)] = self.mu2_rand
                else:
                    B0[self._idx0(2, n2), self._idx0(2, n2 - 1)] = self.mu2_zip

        # --- B0 の対角成分の構築 (各行の和が0になるように設定) ---
        for i in range(size0):
            out_rate = np.sum(B0[i, :]) + np.sum(B_plus1[i, :])
            B0[i, i] = -out_rate

        return B_plus1, B_minus1, B0
    
    def solve_R(self, tol=1e-8, max_iter=10000, verbose=False):
        """
        公比行列 R（最小の非負解）を求める。

        解く二次行列方程式 (論文の式3.12):
            Q_{+1} + R Q_0 + R^2 Q_{-1} = 0

        反復式 (論文の式3.13):
            R_{k+1} = (Q_{+1} + R_k^2 Q_{-1}) (-Q_0)^{-1}
        """
        Q_plus1 = self.Q_plus1
        Q_minus1 = self.Q_minus1
        Q0 = self.Q0

        n = Q0.shape[0]
        R = np.zeros((n, n))
        
        # (-Q_0) の逆行列を事前計算
        inv_neg_Q0 = np.linalg.inv(-Q0)

        for it in range(max_iter):
            R2 = R @ R
            # R_{k+1} = (Q_{+1} + R_k^2 Q_{-1}) * (-Q_0)^{-1}
            R_new = (Q_plus1 + R2 @ Q_minus1) @ inv_neg_Q0

            err = np.max(np.abs(R_new - R))
            R = R_new

            if verbose and (it % 100 == 0 or err < tol):
                print(f"solve_R iter {it}, err={err}")

            if err < tol:
                if verbose:
                    print(f"solve_R converged in {it+1} iterations (err={err})")
                return R

        raise RuntimeError(f"solve_R did not converge in {max_iter} iterations (last err={err})")

if __name__ == "__main__":
    # K=5, m=3, p=0.8 としたテスト
    model = DynamicThresholdQBD(lam1=10, lam2=5, mu1_rand=40, mu2_rand=30, mu1_zip=30, mu2_zip=20, K=5, m=3, p=0.8)
    
    print("Q_minus1 shape:", model.Q_minus1.shape)
    print("B_0 shape:", model.B0.shape)
    print("Q0[0,0] (diagonal check):", model.Q0[0, 0])