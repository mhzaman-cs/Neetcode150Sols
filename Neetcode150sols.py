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

# Medium (12/28/2023): https://leetcode.com/problems/validate-binary-search-tree/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        
        def validRangeBST(root, left, right):
            if not root:
                return True
            if root.val > left and root.val < right:
                return validRangeBST(root.left, left, root.val) and validRangeBST(root.right, root.val, right)
            else:
                return False
        
        return validRangeBST(root, float('-inf'), float('inf'))

# Medium (12/28/2023): https://leetcode.com/problems/kth-smallest-element-in-a-bst/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:


        sorted_array = []

        def returnVals(root):
            if not root:
                return
            returnVals(root.left)
            sorted_array.append(root.val)
            returnVals(root.right)
        
        returnVals(root)

        # print(sorted_array)

        return sorted_array[k-1]

# Medium (12/28/2023): https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        
        def createNodes(preorder, inorder):
            if not preorder:
                return None

            root, inorder_index  = TreeNode(preorder[0]), inorder.index(preorder[0])

            left_inorder = inorder[:inorder_index]
            len_left_inorder = len(left_inorder)+1
            
            root.left = createNodes(preorder[1: len_left_inorder], left_inorder)
            root.right = createNodes(preorder[len_left_inorder:], inorder[inorder_index+1:])
            return root

        return createNodes(preorder, inorder)

# Medium (12/28/2023): https://leetcode.com/problems/number-of-islands/

# DFS:

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        len_c, len_r = len(grid[0]), len(grid)
        num_islands = 0
        visted_coords = set()

        def dfs(r, c):
            if grid[r][c] == "1":
                visted_coords.add((r, c))
                directions = [[0,1], [-1, 0], [0, -1], [1,0]]

                for (inc_r, inc_c) in directions:
                    dr, dc = r+inc_r, c+inc_c 
                    if dr >= 0 and dr < len_r and dc >= 0 and dc < len_c and grid[dr][dc] == "1" and (dr, dc) not in visted_coords:
                        dfs(dr, dc)

        for r in range(len_r):
            for c in range(len_c):
                if grid[r][c] == "1" and (r, c) not in visted_coords:
                    dfs(r, c)
                    num_islands += 1

        return num_islands

# BFS:
    
from collections import deque

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        len_c, len_r = len(grid[0]), len(grid)
        num_islands = 0
        visted_coords = set()

        def bfs(r, c):

            q = deque([(r,c)])

            while q:
                cr, cc = q.popleft()
                directions = [[0, 1], [1, 0], [-1, 0], [0, -1]]
                for inc_r, inc_c in directions:
                    dr, dc = cr + inc_r, cc + inc_c
                    if (dr >= 0 and dr < len_r and 
                    dc >= 0 and dc < len_c and 
                    grid[dr][dc] == "1" and 
                    (dr, dc) not in visted_coords):
                        visted_coords.add((dr, dc))
                        q.append((dr, dc))

        for r in range(len_r):
            for c in range(len_c):
                if grid[r][c] == "1" and (r, c) not in visted_coords:
                    bfs(r, c)
                    num_islands += 1

        return num_islands

# Medium (12/30/2023): https://leetcode.com/problems/clone-graph/

"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        visted_nodes = {}
        def create_graph(node):
            if node in visted_nodes:
                return visted_nodes[node]

            root = Node(node.val)
            visted_nodes[node] = root
            for cur_node in node.neighbors:
                root.neighbors.append(create_graph(cur_node))
            return root
        
        return create_graph(node) if node else None

# Medium (12/30/2023): https://leetcode.com/problems/pacific-atlantic-water-flow/

class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        len_r, len_c = len(heights), len(heights[0])
        visit_pacific = set()
        visit_atlantic = set()

        def dfs(r, c, visted, prevHeight):
            if (r < 0 or r >= len_r or 
            c < 0 or c >= len_c or 
            (r,c) in visted or 
            heights[r][c] < prevHeight):
                return
            visted.add((r,c))
            directions = [(0,1), (1,0), (-1,0), (0,-1)]
            for inc_r, inc_c in directions:
                dfs(r+inc_r, c+inc_c, visted, heights[r][c])


        for r in range(len_r):
            dfs(r, 0, visit_pacific, heights[r][0])
            dfs(r, len_c-1, visit_atlantic, heights[r][len_c-1])
        
        for c in range(len_c):
            dfs(0, c, visit_pacific, heights[0][c])
            dfs(len_r-1, c, visit_atlantic, heights[len_r-1][c])
        
        results = []

        for r in range(len_r):
            for c in range(len_c):
                if (r, c) in visit_pacific and (r,c) in visit_atlantic:
                    results.append([r, c])

        return results

# Medium (12/30/2023): https://leetcode.com/problems/course-schedule/

class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        preReqMap = {i:[] for i in range(numCourses)}

        for crs, preq in prerequisites:
            preReqMap[crs].append(preq)

        def dfs(crs, visted):
            if crs in visted:
                return False
            if preReqMap[crs] == []:
                return True
            visted.add(crs)

            for crs_preq in preReqMap[crs]:
                if not dfs(crs_preq, visted):
                    return False
            preReqMap[crs] = []
            visted.remove(crs)
            return True

        for c in range(numCourses):
            if not dfs(c, set()):
                return False

        return True

# Medium (12/30/2023): https://www.lintcode.com/problem/3651/

class Solution:
    def count_components(self, n: int, edges: List[List[int]]) -> int:
        parents = [i for i in range(n)]
        rank = [1] * n
        cur_nodes = n

        def find(node):
            cur_node = node
            while cur_node != parents[cur_node]:
                parents[cur_node] = parents[parents[cur_node]]
                cur_node = parents[cur_node]
            return cur_node

        def union(node1, node2):
            p1, p2 = find(node1), find(node2)
            if p1 == p2:
                return 0
            if rank[p1] >= rank[p2]:
                rank[p1] += rank[p2] 
                parents[p2] = p1
            else:
                rank[p2] += rank[p1] 
                parents[p1] = p2
            return 1

        for a, b in edges:
            cur_nodes -= union(a, b)

        return cur_nodes

# Medium (12/31/2023): https://leetcode.com/problems/number-of-provinces/

class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        num_cities = len(isConnected[0])
        parents = [i for i in range(num_cities)]
        rank = [1] * num_cities
        num_cities_left = num_cities

        def find(node):
            cur_node = node
            while cur_node != parents[cur_node]:
                parents[cur_node] = parents[parents[cur_node]]
                cur_node = parents[cur_node]
            
            return cur_node

        def union(n1, n2):
            p1, p2 = find(n1), find(n2)

            if p1 == p2:
                return 0
            
            if rank[p1] >= rank[p2]:
                rank[p1] += rank[p2]
                parents[p2] = p1
            else:
                rank[p2] += rank[p1]
                parents[p1] = p2

            return 1

        for i in range(num_cities):
            for j in range(num_cities):
                if isConnected[i][j] == 1:
                    num_cities_left -= union(i, j)

        return num_cities_left

# Medium (1/1/2024): https://www.lintcode.com/problem/178/

from typing import (
    List,
)

class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def valid_tree(self, n: int, edges: List[List[int]]) -> bool:
        edges_map = {i:[] for i in range(n)}
        if not edges:
            return n == 1
        for a,b in edges:
            edges_map[a].append(b)
            edges_map[b].append(a)

        visted_set = set()
        def dfs(node, prevVal):
            if node in visted_set:
                return False
            visted_set.add(node)
            for nei in edges_map[node]:
                if prevVal == nei:
                    continue
                if not dfs(nei, node):
                    return False
            return True

        if not dfs(0, -1):
            return False

        if len(visted_set) == n:
            return True
        return False 

# Medium (1/1/2024): https://leetcode.com/problems/insert-interval/

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.append([float('inf'), float('inf')])
        s, e = newInterval
        len_inter = len(intervals)
        for i in range(len_inter):
            if e < intervals[i][0]:
                if i == 0 or intervals[i-1][1] < s:
                    return intervals[:i] + [newInterval] + intervals[i:len_inter-1]               
                d = i-1
                while (d > 0 and s <= intervals[d-1][1]):
                    d -= 1
                return intervals[:d] + [[min(intervals[d][0], s), max(e, intervals[i-1][1])]] + intervals[i:len_inter-1]

# Medium (1/1/2024): https://leetcode.com/problems/merge-intervals/

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x : x[0])
        len_intervals = len(intervals)
        res = []
        cur_interval = intervals[0]
        for i in range(1, len_intervals):
            if cur_interval[1] < intervals[i][0]:
                res.append(cur_interval)
                cur_interval = intervals[i]
            else:
                cur_interval = [min(cur_interval[0], intervals[i][0]), max(cur_interval[1], intervals[i][1])]
        res.append(cur_interval)
        return res

# Medium (1/1/2024): https://leetcode.com/problems/non-overlapping-intervals/

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort()
        cur_removed = 0
        curr_end = intervals[0][1]
        intervals.pop(0)
        for start, end in intervals:
            if start < curr_end:
                cur_removed += 1
                curr_end = min(end, curr_end)
            else:
                curr_end = end

        return cur_removed

# Medium (1/1/2024): https://www.lintcode.com/problem/919/

from typing import (
    List,
)
from lintcode import (
    Interval,
)

"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    def min_meeting_rooms(self, intervals: List[Interval]) -> int:
        intervals.sort(key=lambda x : x.start)
        end_times = [intervals[0].end]
        intervals.pop(0)

        for i in intervals:
            start, end = i.start, i.end
            if start >= min(end_times):
                end_times.remove(min(end_times))
                end_times.append(end)
            else:
                end_times.append(end)
        return len(end_times)

# Medium (1/1/2024): https://leetcode.com/problems/house-robber/

# Memoization
class Solution:
    def rob(self, nums: List[int]) -> int:
        memoziation = {}
        len_array = len(nums)
        if len_array == 0:
            return 0
        if len_array == 1:
            return nums[0]
        if len_array == 2:
            return max(nums[0], nums[1])

        memoziation[len_array-1] = nums[len_array-1]
        memoziation[len_array-2] = max(nums[len_array-1], nums[len_array-2])
        
        for i in range(len_array-3, -1, -1):
            memoziation[i] = max(nums[i] + memoziation[i+2], memoziation[i+1])

       return memoziation[0]

# Bottom Up
class Solution:
    def rob(self, nums: List[int]) -> int:
        h1, h2, len_nums = 0, 0, len(nums)
        for n in nums:
            h2, h1 = max(n+h1, h2), h2

        return h2

# Medium (1/1/2024): https://leetcode.com/problems/house-robber/

class Solution:
    def rob(self, nums: List[int]) -> int:
        len_array = len(nums)

        def house_robber(nums):
            h1 = h2 = 0

            for n in nums:
                h2, h1 = max(h1+n, h2), h2

            return h2

        return max(nums[0], house_robber(nums[1:]), house_robber(nums[:-1]))

# Medium (1/1/2024): https://leetcode.com/problems/longest-palindromic-substring/

class Solution:
    def longestPalindrome(self, s: str) -> str:
        len_s = len(s)
        max_len = 0
        max_pal = ""

        for c in range(len_s):
            # Odd case
            l, r = c, c

            while (l >=0 and r < len_s  and s[l] == s[r]):
                if (r -l + 1) > max_len:
                    max_len = r -l + 1
                    max_pal = s[l:r+1]
                l-=1
                r+=1
                
            l, r = c, c+1

            # Even Case
            while (l >=0 and r < len_s  and s[l] == s[r]):
                if (r -l + 1) > max_len:
                    max_len = r -l + 1
                    max_pal = s[l:r+1]
                l-=1
                r+=1
            
        return max_pal

# Medium (1/1/2024): https://leetcode.com/problems/palindromic-substrings/

class Solution:
    def countSubstrings(self, s: str) -> int:

        num_subtrings = 0
        len_s = len(s)

        for c in range(len_s):
            l = r = c

            while l>=0 and r<len_s:
                if s[l] == s[r]:
                    num_subtrings += 1
                else:
                    break
                l-= 1
                r+=1

            l, r = c, c+1

            while l>=0 and r<len_s:
                if s[l] == s[r]:
                    num_subtrings += 1
                else:
                    break
                l-= 1
                r+=1
        return num_subtrings

# Medium (1/2/2024): https://leetcode.com/problems/decode-ways/

class Solution:
    def numDecodings(self, s: str) -> int:
        one_d, two_d = 1, 0

        len_s = len(s)

        for i in range(len_s-1, -1, -1):
            cur_val = 0
            if s[i] == '0':
                cur_val = 0
            else:
                cur_val = one_d

            if (i+1 < len_s and (s[i] == '1' or (s[i] == '2' and s[i+1] in "0123456"))):
                cur_val += two_d

            one_d, two_d = cur_val, one_d
        return one_d

# Medium (1/2/2024): https://leetcode.com/problems/coin-change/

# Memoization Solution
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:

        num_coins = {x:1 for x in coins}
        def minRequiredCoins(amount):
            if amount == 0:
                return 0
            if amount < 0:
                return float('inf')
            if amount in num_coins:
                return num_coins[amount]
            min_coins = float('inf')
            for c in coins:
                min_coins = min(min_coins, 1+minRequiredCoins(amount - c))
            num_coins[amount] = min_coins
            return min_coins

        res =  minRequiredCoins(amount)
        return res if res != float('inf') else -1

# Dp - backtracking
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount+1)
        dp[0] = 0
        for i in range(1, amount+1):
            min_coins = dp[i]
            for c in coins:
                if i-c >= 0:
                    min_coins = min(min_coins, 1+dp[i-c])
            dp[i] = min_coins
        
        return dp[-1] if dp[-1] != float('inf') else -1

# Medium (1/2/2024): https://leetcode.com/problems/maximum-product-subarray/

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        cur_max, max_pos, max_neg = max(nums), 1, 1

        for n in nums:
            max_neg, max_pos = min(n, n * max_pos, n * max_neg), max(n, n * max_pos, n * max_neg)
            cur_max = max(max_pos, cur_max)
 
        return cur_max

# Medium (1/2/2024): https://leetcode.com/problems/word-break/

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        len_s = len(s)
        dp = [False] * (len_s+1)
        dp[len_s] = True

        for i in range(len_s-1, -1, -1):
            for cur_word in wordDict:
                if (i+ len(cur_word)) <= len_s and s[i:i+len(cur_word)] == cur_word:
                    dp[i] = dp[i+len(cur_word)]

                if dp[i]:
                    break
        return dp[0]

# Medium (1/2/2024): https://leetcode.com/problems/longest-increasing-subsequence/

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        len_nums = len(nums)
        dp = [1] * len_nums

        for i in range(len_nums-1, -1, -1):
            for j in range(i+1, len_nums):
                if nums[j] > nums[i]:
                    dp[i] = max(dp[i], 1+dp[j])
        return max(dp)

# Medium (1/3/2024): https://leetcode.com/problems/unique-paths/

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        row = [1] * n

        for i in range(m-1):
            for j in range(n-2, -1, -1):
                row[j] = row[j + 1] + row[j]

        return row[0]

# Medium (1/3/2024): https://leetcode.com/problems/longest-common-subsequence/

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len_1, len_2 = len(text1), len(text2)
        dp = [[0] * (len_1+1) for _ in range(len_2+1)]

        for i in range(len_1-1, -1, -1):
            for j in range(len_2-1, -1, -1):
                if text1[i] == text2[j]:
                    dp[j][i] = 1+dp[j+1][i+1]
                else:
                    dp[j][i] = max(dp[j+1][i], dp[j][i+1])

        return dp[0][0]

# Medium (1/3/2024): https://leetcode.com/problems/maximum-subarray/

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        l, len_nums = 0, len(nums)
        max_sum = cur_sum = nums[0]
        for r in range(1, len_nums):
            if cur_sum < 0:
                l = r
                cur_sum = nums[r]
            else:
                cur_sum += nums[r]
            max_sum = max(cur_sum, max_sum)

        return max_sum

# Medium (1/3/2024): https://leetcode.com/problems/jump-game/

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        goal = len(nums) -1

        for i in range(len(nums)-2, -1, -1):
            if i+nums[i] >= goal:
                goal  = i
        return goal == 0

# Medium (1/3/2024): https://leetcode.com/problems/rotate-image/

class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """

        l, r = 0, len(matrix)-1

        while l < r:
            for i in range(r -l):
                t, b = l, r
                matrix[t][l+i], matrix[t+i][r], matrix[b][r-i], matrix[b-i][l] = matrix[b-i][l], matrix[t][l+i], matrix[t+i][r], matrix[b][r-i]
            r -= 1
            l += 1

# Medium (1/5/2024): https://leetcode.com/problems/spiral-matrix/

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:

        result = []
        t = l = 0
        r = len(matrix[0])       
        b = len(matrix)

        while l < r and t < b:
            for i in range(l, r):
                result.append(matrix[t][i])
            t += 1

            for i in range(t, b):
                result.append(matrix[i][r-1])
            r -= 1

            if not (l < r and t < b):
                break
            
            for i in range(r-1, l-1, -1):
                result.append(matrix[b-1][i])
            b -= 1


            for i in range(b-1, t-1, -1):
                result.append(matrix[i][l])
            l += 1

        return result

# Medium (1/5/2024): https://leetcode.com/problems/set-matrix-zeroes/

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        zeros_r = set()
        zeros_c = set()
        r, c = len(matrix), len(matrix[0])
        for i in range(r):
            for j in range(c):
                if matrix[i][j] == 0:
                   zeros_r.add(i)
                   zeros_c.add(j)

        for cr in zeros_r:
            for k in range(c):
                matrix[cr][k] = 0

        for cc in zeros_c:
            for k in range(r):
                matrix[k][cc] = 0
        