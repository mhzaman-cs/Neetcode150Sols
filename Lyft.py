# Hard (2/27/2024): https://leetcode.com/problems/minimum-window-substring/

from collections import defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:

        if len(s) < 2:
            return s if t in s else ""
        l, r = 0, 0
        t_freq = defaultdict(int)
        s_freq = defaultdict(int)
        shortest_l, shortest_r = float('-inf'), float('inf')

        for c in t:
            t_freq[c] = t_freq.get(c, 0) + 1
        
        num_unique = len(t_freq)
        while (r < len(s) or (r == len(s) and num_unique == 0)):
            if num_unique == 0:
                if (shortest_r - shortest_l) > (r-l):
                    shortest_l, shortest_r = l, r

                s_freq[s[l]] -= 1
                num_unique += t_freq[s[l]] > s_freq[s[l]]
                l += 1
            else:
                s_freq[s[r]] = s_freq.get(s[r], 0) + 1
                if s_freq[s[r]] == t_freq[s[r]]:
                    num_unique -= 1
                r += 1
        return s[shortest_l: shortest_r] if shortest_l != float('-inf') else ""

        