# Medium (1/19/2024): https://leetcode.com/problems/flatten-nested-list-iterator/

# BFS solution

from collections import deque
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        # print(nestedList)
        nestedDeque = deque(nestedList)
        self.nestedList = deque([])
        while nestedDeque:
            leftNode = nestedDeque.popleft()
            if leftNode.isInteger():
                self.nestedList.append(leftNode.getInteger())
            else:
                len_leftNode = len(leftNode.getList())
                for i in range(len_leftNode-1, -1, -1):
                    nestedDeque.appendleft(leftNode.getList()[i])





    
    def hasNext(self) -> bool:
        return len(self.nestedList) != 0
    
    def next(self) -> int:
        return self.nestedList.popleft()
         

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())
    

# Time complexity: O(N)
    
# BFS solution

# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """
#
#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """
#
#    def getList(self) -> [NestedInteger]:
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """

class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):

        self.nestedList = []
        def flatten_list(nestedList):
            for current_item in nestedList:
                if current_item.isInteger():
                    self.nestedList.append(current_item.getInteger())
                else:
                    flatten_list(current_item.getList())

        flatten_list(nestedList)
        self.nestedList.reverse() 
                
    
    def hasNext(self) -> bool:
        return self.nestedList
    
    def next(self) -> int:
        return self.nestedList.pop()
         

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())