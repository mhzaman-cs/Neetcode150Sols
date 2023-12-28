# Easy (12/4/2023): https://leetcode.com/problems/contains-duplicate/

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        freq_map = {}
        for cur_num in nums:
            if cur_num in freq_map:
                return True
            freq_map[cur_num] = 1
        return False 
        

# Easy (12/5/2023): https://leetcode.com/problems/valid-anagram/

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        lenS, lenT =  len(s), len(t)

        if (lenS != lenT): return False 
        
        wordCountS = {}
        wordCountT = {}

        for i in range(lenS):
            if s[i] not in wordCountS.keys():
                wordCountS[s[i]] = 0
            if t[i] not in wordCountT.keys():
                wordCountT[t[i]] = 0

            wordCountS[s[i]] += 1
            wordCountT[t[i]] += 1

        return wordCountS == wordCountT

# Easy (12/5/2023): https://leetcode.com/problems/two-sum/

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        needatIndex = {}
        lenNums = len(nums)
        for i in range(lenNums):
            num = nums[i]
            cur_diff = target - num
            if num in needatIndex:
                return [i, needatIndex[num]]
            needatIndex[cur_diff] = i

# Easy (12/5/2023): https://leetcode.com/problems/valid-palindrome/

class Solution:
    def isPalindrome(self, s: str) -> bool:
        lenS = len(s)
        l, r =  0, lenS - 1
        while (r >= l and r >= 0 and r <= lenS and l >= 0 and l <= lenS):
            if (s[l].isalnum()  and s[r].isalnum()):
                if s[l].lower() == s[r].lower():
                    l += 1
                    r -= 1
                else:
                    return False
            elif (not (s[l].isalnum())):
                l += 1
            elif (not (s[r].isalnum())):
                r -= 1

        return True

# Easy (12/5/2023): https://leetcode.com/problems/valid-parentheses/

class Solution:
    def isValid(self, s: str) -> bool:
        inputStack = []
        OpeningBrackets = {'(' : ')', '{' : '}', '[' : ']'}

        for i in s:
            if i in OpeningBrackets:
                inputStack.append(i)
            else:
                if inputStack and OpeningBrackets[inputStack[-1]] == i:
                    inputStack.pop()
                else:
                    return False
        return not inputStack

# Easy (12/5/2023): https://leetcode.com/problems/binary-search/

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        if not nums:
            return -1
        if (target == nums[0]):
            return 0
        while (l < r):
            mid = (l + r) // 2
            cur_val = nums[mid]
            if (target > cur_val):
                l = mid + 1
            elif (target < cur_val):
                r = mid
            else:
                return mid

        if (l == r and nums[l] == target):
            return l
        return -1


# Easy (12/6/2023): https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l, r = 0, 1
        maxProfit =  0
        lenPrices = len(prices) 

        while (r < lenPrices):
            maxProfit = max(maxProfit, prices[r] - prices[l])
            if (prices[r] < prices[l]):
                l, r = r, r+1
            else:
                r+=1

        return maxProfit

# Easy (12/6/2023): https://leetcode.com/problems/reverse-linked-list/

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prevNode = None
        cur_node=head
        while cur_node:
           tempNode = cur_node.next
           cur_node.next = prevNode
           prevNode = cur_node
           cur_node = tempNode
        return prevNode

# Not accepted solution, but one with accumulator variable
def linkListRev(curNode, prevNode):
  if curNode == None:
    return prevNode
  tempNode = curNode.next
  curNode.next = prevNode
  linkListRev(tempNode, curNode)


class Solution:
  def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
    if not head:
      return None
    newHead = head
    if head.next:
      newHead = self.reverseList(head.next)
      head.next.next = head
    head.next = None
    return newHead

# Easy (12/7/2023): https://leetcode.com/problems/merge-two-sorted-lists/

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
      head = ListNode()
      tail = head 
      
      while list1 and list2:
        if list2.val <= list1.val:
          tail.next = list2
          list2 = list2.next
        else:
          tail.next = list1
          list1 = list1.next
        tail = tail.next

      if list1:
        tail.next = list1 
      if list2:
        tail.next = list2
      
      return head.next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
      if not list1:
        return list2
      if not list2:
        return list1

      if (list1.val < list2.val):
        list1.next = self.mergeTwoLists(list1.next, list2)
        return list1
      else:
        list2.next = self.mergeTwoLists(list1, list2.next)
        return list2

# Easy (12/7/2023): https://leetcode.com/problems/linked-list-cycle/

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head or not head.next:
          return False 
        f, s = head, head
        while (s and f):
            if (s.next and f.next and f.next.next):
                s, f = s.next, f.next.next
            else:
                return False
            if s == f:
                return True

        return False

# Easy (12/7/2023): https://leetcode.com/problems/invert-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root or (not root.left and not root.right):
            return root
        
        tempNode = root.left
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(tempNode)
        return root

# Easy (12/7/2023): https://leetcode.com/problems/maximum-depth-of-binary-tree/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0

        return max(1 + self.maxDepth(root.right), 1 + self.maxDepth(root.left))

# Easy (12/8/2023): https://leetcode.com/problems/diameter-of-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_diam = 0

        def dfs(root):
            nonlocal max_diam
            if not root:
                return 0

            left = dfs(root.left)
            right = dfs(root.right)

            max_diam = max(left + right, max_diam)

            return 1 + max(left, right)

        dfs(root)
        return max_diam

# Easy (12/9/2023): https://leetcode.com/problems/balanced-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        isHeightBalenced = True

        def treeHeight(root):
            nonlocal isHeightBalenced
            if not root:
                return 0

            leftHeaight = treeHeight(root.left)
            rightHeaight = treeHeight(root.right)

            if abs(leftHeaight - rightHeaight) > 1 or not isHeightBalenced:
                isHeightBalenced = False
                return 0

            return 1 + max(leftHeaight, rightHeaight)
        
        treeHeight(root)

        return isHeightBalenced

# Easy (12/9/2023): https://leetcode.com/problems/same-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True

        if not p or not q:
            return False
        
        return (p.val == q.val) and self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)

# Easy (12/9/2023): https://leetcode.com/problems/subtree-of-another-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not subRoot:
            return True
        if not root and subRoot:
            return False
        return self.isSameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)  
    
    def isSameTree(self, root, subRoot):
        if not root and not subRoot:
            return True
        if not root or not subRoot:
            return False
        return root.val == subRoot.val and self.isSameTree(root.right, subRoot.right) and self.isSameTree(root.left, subRoot.left)

# Easy (12/9/2023): https://leetcode.com/problems/climbing-stairs/
class Solution:
    def climbStairs(self, n: int) -> int:
        one = two = 1

        for i in range(n-1):
            one, two = one + two, one

        return one

# Easy (12/9/2023): https://leetcode.com/problems/meeting-rooms/
# Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end


class Solution:
    def can_attend_meetings(self, intervals: List[Interval]) -> bool:
        lenItervals = len(intervals)
        intervals.sort(key = lambda x: x.start)
        for i in range(lenItervals - 1):
            if intervals[i].end > intervals[i+1].start:
                return False
        return True

# Easy (12/9/2023): https://leetcode.com/problems/number-of-1-bits/

class Solution:
    def hammingWeight(self, n: int) -> int:
        curCount = 0
        for i in str(bin(n))[2:]:
            if i == '1':
                curCount+=1
        return curCount

class Solution:
    def hammingWeight(self, n: int) -> int:
        curCount = 0
        while (n > 0):
            curCount += n & 1
            n = n >> 1
        return curCount

# Easy (12/10/2023): https://leetcode.com/problems/counting-bits/

class Solution:
    def countBits(self, n: int) -> List[int]:
        res = []
        for i in range(n+1):
            curCount = 0
            k = i
            while k:
                curCount += k & 1
                k = k >> 1
            res.append(curCount)

        return res

class Solution:
    def countBits(self, n: int) -> List[int]:
        mostSig = 1
        result = [0] * (n+1)

        if n >= 0:
            result[0] = 0
        if n >= 1:
            result[1] = 1

        for i in range(2, n+1):
            if mostSig*2 == i:
                mostSig = i
            result[i] = 1 + result[i - mostSig]

        return result

# Easy (12/10/2023): https://leetcode.com/problems/reverse-bits/

class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            res = res | (((n >> i) & 1) << (31-i))
        return res

# Easy (12/10/2023): https://leetcode.com/problems/missing-number/

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        return sum([x for x in range(len(nums) + 1)]) - sum(nums)

class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        r_nums = r_range = 0
        for i in nums:
            r_nums = r_nums ^ i
        
        for i in range(n+1):
            r_range = r_range ^ i
        return r_nums ^ r_range

# Medium (12/10/2023): https://leetcode.com/problems/group-anagrams/

class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hash_map_array = []
        result_array = []

        for word in strs:
            char_count = {}
            for cur_char in word:
                char_count[cur_char] = char_count.get(cur_char, 0) + 1
            if (char_count in hash_map_array):
                result_array[hash_map_array.index(char_count)].append(word)
            else:
                hash_map_array.append(char_count)
                result_array.append([word])

        return result_array

# Medium (12/11/2023): https://leetcode.com/problems/top-k-frequent-elements/

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq_map = {}
        for n in nums:
            freq_map[n] = freq_map.get(n, 0) + 1
        

        sorted_freq_map = sorted(freq_map.items(), key=lambda x:-x[1])

        result = []
        num_count = 0
        for num in sorted_freq_map:
            if k  - num_count == 0:
                break 
            result.append(num[0])
            num_count+=1

        return result

# Medium (12/11/2023): https://leetcode.com/problems/product-of-array-except-self/

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        range_vals = {}
        lenNums = len(nums)
        res = [] * lenNums
        cur_product_f = 1
        for i in range(lenNums):
            cur_product_f *= nums[i]
            range_vals[(0, i)] = cur_product_f

        cur_product_b = 1
        for i in range(lenNums-1, -1, -1):
            cur_product_b *= nums[i]
            range_vals[(i, lenNums-1)] = cur_product_b

        for i in range(lenNums):
            if i == 0:
                res.append(range_vals[(min(lenNums-1, i+1), lenNums-1)])
            elif i == lenNums-1:
                res.append(range_vals[(0, max(i-1, 0))])
            else:
                res.append(range_vals[(0, max(i-1, 0))] * range_vals[(min(lenNums-1, i+1), lenNums-1)])

        return res

# Medium (12/12/2023): https://www.lintcode.com/problem/659/

class Solution:
    def encode(self, strs):
        aux_str = ""
        for word in strs:
            aux_str += str(len(word))+ "%" + word
        return aux_str

    def decode(self, str):
        res = []
        i, lenStr = 0, len(str)
        lenJump, lenJumpStr = 0, ""

        while i < lenStr:
            if str[i] == "%":
                lenJump = int(lenJumpStr)
                res.append(str[i+1:i+1+lenJump])
                i = lenJump+i+1
                lenJumpStr = ""
            else:
                lenJumpStr += str[i]
                i+=1
        return res

# Medium (12/12/2023): https://leetcode.com/problems/longest-consecutive-sequence/

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        set_nums = set(nums)
        max_cons = 0

        for n in set_nums:
            if n-1 not in set_nums:
                cur_num = n+1
                cur_cons = 1
                while cur_num in set_nums:
                    cur_cons += 1
                    cur_num += 1
                max_cons = max(max_cons, cur_cons)
        
        return max_cons

# Medium (12/13/2023): https://leetcode.com/problems/3sum/

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res = []
        i = 0
        lenNums = len(nums)
        while i < lenNums and nums[i] <= 0:
            if i+1 >= lenNums-1:
                break
            l, r = i+1, lenNums-1
            while l < r:
                val_i, val_r, val_l = nums[i], nums[r], nums[l]
                cur_sum = val_i + val_r + val_l
                if cur_sum == 0:
                    if [nums[i], nums[l], nums[r]] not in res:
                        res.append([nums[i], nums[l], nums[r]])
                    r -= 1
                    l += 1
                elif cur_sum > 0:
                    r -= 1
                else:
                    l += 1
            i += 1

            
        return res

# Medium (12/13/2023): https://leetcode.com/problems/container-with-most-water/

class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height)-1
        maxVol = 0
        while l < r:
            maxVol = max((r - l) * min(height[l], height[r]), maxVol)

            if height[l] < height[r]:
                l+=1
            else:
                r-=1
        return maxVol

# Medium (12/14/2023): https://leetcode.com/problems/longest-substring-without-repeating-characters/

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        lenS = len(s)
        if lenS <= 1:
            return lenS 

        maxSLen = 0
        l, r = 0, 1
        while l < r and r < lenS:
            curStr = s[l:r]
            lenCurS = len(curStr)
            if s[r] not in curStr:
                r += 1
            else:
                maxSLen = max(lenCurS, maxSLen)
                while s[r] in curStr and l < r:
                    l += 1
                    curStr = s[l:r]
                r +=1
        
        maxSLen = max(len(s[l:r]), maxSLen)
        
        return maxSLen

# Medium (12/21/2023): https://leetcode.com/problems/longest-repeating-character-replacement/

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        freq_map = {}
        max_freq = 1
        max_len = 1
        lenS = len(s)
        l = 0

        for r in range(lenS):
            freq_map[s[r]] = freq_map.get(s[r], 0) + 1
            max_freq = max(max_freq, freq_map[s[r]])

            if (r - l + 1) - max_freq > k:
                freq_map[s[l]] -= 1
                l += 1
            
            max_len = max(max_len, r-l+1)

        return max_len

# Medium (12/21/2023): https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/

class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums) - 1
        cur_min= nums[0]
        while l < r:
            mid = (l+r)//2
            cur_min=min(nums[mid], cur_min)
            
            if nums[mid] > nums[r]:
                l = mid + 1
            else:
                r = mid - 1
        return min(cur_min, nums[l])

# Medium (12/22/2023): https://leetcode.com/problems/search-in-rotated-sorted-array/

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l, r = 0, len(nums)-1
        not_found = -1
        while l <= r:
            mid = (l+r)//2
            if nums[mid] == target:
                return mid
            elif ((nums[r] < nums[mid] or target > nums[mid]) and nums[r] >= target) or (target > nums[mid] and nums[r] < target and nums[mid] > nums[l]):
                l = mid + 1
            else:
                r = mid - 1
        return not_found
        
# Medium (12/25/2023): https://leetcode.com/problems/reorder-list/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        cur_p = f = s = head

        while f and f.next:
            f = f.next.next
            s = s.next
        s.next, s = None, s.next
        
        prev_p = None
        while s:
            s.next, prev_p, s = prev_p, s, s.next
        s = prev_p
        
        while s:
            cur_p.next, s.next, s, cur_p = s, cur_p.next, s.next, cur_p.next

# Medium (12/25/2023): https://leetcode.com/problems/remove-nth-node-from-end-of-list/

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        cur_p = end_p = head

        while n >= 0:
            if not end_p:
                return head.next
            end_p = end_p.next
            n -= 1
        
        while end_p:
            cur_p, end_p = cur_p.next, end_p.next
        
        cur_p.next = cur_p.next.next 
        return head

# Medium (12/27/2023): https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        cur_p = root

        while cur_p:
            if (p.val > cur_p.val) and (q.val > cur_p.val):
                cur_p = cur_p.right
            elif (p.val < cur_p.val) and (q.val < cur_p.val):
                cur_p = cur_p.left
            else:
                return cur_p


# Medium (12/28/2023): https://leetcode.com/problems/binary-tree-level-order-traversal/

from collections import deque

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        
        results = [[]]
        treeD, num_treeD_level = deque([root]), 1

        while treeD:
            cur_n = treeD.popleft()
            num_treeD_level -= 1

            if cur_n.left: 
                treeD.append(cur_n.left)
            if cur_n.right: 
                treeD.append(cur_n.right)
            
            results[-1].append(cur_n.val)
            
            if num_treeD_level == 0 and len(treeD) != 0:
                results.append([])
                num_treeD_level = len(treeD)
        return results
        