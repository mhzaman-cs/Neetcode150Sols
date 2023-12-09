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

# Easy (12/7/2023): https://leetcode.com/problems/diameter-of-binary-tree/

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

        