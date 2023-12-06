# Easy (12/4/2023): https://leetcode.com/problems/contains-duplicate/

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        freq_map = {}
        for cur_num in nums:
            if cur_num in freq_map:
                return True
            freq_map[cur_num] = 1
        return False 
        

# Easy (12/5/2023): https://leetcode.com/problems/valid-anagram/submissions/

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

# Easy (12/5/2023): https://leetcode.com/problems/two-sum/submissions/

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

# Easy (12/5/2023): https://leetcode.com/problems/valid-palindrome/submissions/

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

# Easy (12/5/2023): https://leetcode.com/problems/binary-search/submissions/

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


