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

# Hard (2/27/2024): https://www.lintcode.com/problem/660/solution

class Solution:

    # @param {char[]} buf destination buffer
    # @param {int} n maximum number of characters to read
    # @return {int} the number of characters read
    def __init__(self):
        self.cur_buf, self.i4, self.n4 = [None] * 4, 0 , 0 
    
    def read(self, buf, n):
        i = 0
        while (i < n):
            if self.i4 == self.n4:
                self.i4, self.n4 = 0, Reader.read4(self.cur_buf)
                if not self.n4:
                    break
            buf[i], i, self.i4  = self.cur_buf[self.i4], i+1, self.i4+1  
        return i

# Hard (2/27/2024): https://leetcode.com/problems/range-sum-query-2d-immutable/

class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        self.sumMatrix = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]

        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                self.sumMatrix[j][i] = (self.sumMatrix[j-1][i] if (j -1 >=0) else 0) + matrix[j][i]+ (self.sumMatrix[j][i-1] if i-1 >=0 else 0) - (self.sumMatrix[j-1][i-1] if (i-1 >=0 and j-1>=0) else 0)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        left_square = self.sumMatrix[row2][col1-1] if col1-1 >= 0 else 0
        top_square = (self.sumMatrix[row1-1][col2] if row1-1 >= 0 else 0) - (self.sumMatrix[row1-1][col1-1] if (row1-1 >= 0 and col1-1>=0) else 0)
        return (self.sumMatrix[row2][col2] - top_square) - left_square


# Medium (2/27/2024): https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/

# Your NumMatrix object will be instantiated and called as such:
# obj = NumMatrix(matrix)
# param_1 = obj.sumRegion(row1,col1,row2,col2)

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)

        mid = self.splitList(head)
        treeMid = TreeNode(mid.val)
        treeMid.right = self.sortedListToBST(mid.next)
        treeMid.left = self.sortedListToBST(head)

        return treeMid

    def splitList(self, head):
        dummy = ListNode(0, head)

        p, s, f = dummy, head, head

        while f and f.next:
            p, s, f = p.next, s.next, f.next.next

        p.next = None
        return s

# Medium (2/28/2024): https://leetcode.com/problems/time-based-key-value-store/

from collections import defaultdict
class TimeMap:

    def __init__(self):
        self.TBKV = defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.TBKV[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        values = self.TBKV[key]
        res = ""

        l, r = 0, len(values)-1

        while l <= r:
            mid = (l+r)//2
            if values[mid][0] <= timestamp:
                res = values[mid][1]
                l = mid+1
            else:
                r = mid-1

        return res
        


# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)
    

# Medium (2/28/2024): https://leetcode.com/problems/rotting-oranges/

from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        num_fresh = 0
        rotten = deque([])
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    num_fresh+=1
                elif grid[i][j] == 2:
                    rotten.append((i,j))

        time = 0
        while rotten and num_fresh > 0:

            for _ in range(len(rotten)):
                i, j = rotten.popleft()
                directions = [(0,1), (0,-1), (1,0), (-1,0)]
                for inc_x, inc_y in directions:
                    dx, dy = i+inc_x, j+inc_y

                    if dx>=0 and dx < len(grid) and dy >= 0 and dy < len(grid[0]) and grid[dx][dy] == 1:
                        grid[dx][dy] = 2
                        rotten.append((dx, dy))
                        num_fresh-=1

            time += 1
        return time if not num_fresh else -1
    
from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:

        num_fresh = 0
        rotten = deque([])
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    num_fresh+=1
                elif grid[i][j] == 2:
                    rotten.append((i,j,1))

        time = 0
        while rotten:
            i, j, t = rotten.popleft()
            directions = [(0,1), (0,-1), (1,0), (-1,0)]
            for inc_x, inc_y in directions:
                dx, dy = i+inc_x, j+inc_y

                if dx>=0 and dx < len(grid) and dy >= 0 and dy < len(grid[0]) and grid[dx][dy] == 1:
                    grid[dx][dy] = 2
                    time = max(time, t)
                    rotten.append((dx, dy, t+1))
                    num_fresh-=1

        return time if not num_fresh else -1