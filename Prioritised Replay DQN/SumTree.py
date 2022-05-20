import numpy as np


class Tree:
    def __init__(self,capacity):
        self.capacity = capacity # 葉子數量
        self.tree = np.zeros(self.capacity * 2 - 1) # 總節點數量
        self.start = capacity - 1 # 最左邊的葉子節點index

    def add(self,p,data):
        leaf_idx = self.start + p - 1

        if leaf_idx > len(self.tree) - 1:
            raise ValueError('Leaf index big then total leaf capacity.')

        self._update_tree(leaf_idx, data)

    def get_tree(self):
        return self.tree

    def get_leaf(self,rand):
        p_idx = 0
        while True:
            L_idx = p_idx * 2 + 1
            R_idx = L_idx + 1
            if L_idx >= len(self.tree):
                break
            else:
                if rand <= self.tree[L_idx]:
                    p_idx = L_idx
                else:
                    rand -= self.tree[L_idx]
                    p_idx = R_idx

        return p_idx, self.tree[p_idx]

    def get_total(self):
        return self.tree[0]

    def _update_tree(self, leaf_idx, data):
        change = data - self.tree[leaf_idx]
        self.tree[leaf_idx] = data
        while leaf_idx != 0:
            leaf_idx = (leaf_idx - 1) // 2
            self.tree[leaf_idx] += change

    def batch_update(self,leaf_idx,abs_error):
        for l_idx, new_p in zip(leaf_idx, abs_error):
            self._update_tree(l_idx,new_p)



