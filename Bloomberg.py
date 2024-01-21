# Medium (1/20/2024): https://leetcode.com/problems/invalid-transactions/
class Solution:
    def invalidTransactions(self, transactions: List[str]) -> List[str]:
        transaction_map = {}
        invalid_transactions = []
        for item in transactions:
            name, time, amount, city = item.split(",")

            added = False
            if int(amount) > 1000:
                invalid_transactions.append(item)
                added = True
            transaction_map[name] = transaction_map.get(name, []) + [[time, city, amount, added]]

        for k, v in transaction_map.items():
            v.sort()
            print(v)
            vleng = len(v)
            for i in range(vleng):
                for j in range(i+1, vleng):
                    if j < vleng and v[i][1] != v[j][1] and abs(int(v[i][0]) - int(v[j][0])) <= 60:
                        if not v[i][3]:
                            invalid_transactions.append(k + ',' + v[i][0] + ',' + v[i][2] + ',' + v[i][1])
                            v[i][3] = True
                        if not v[j][3]:
                            invalid_transactions.append(k + ',' + v[j][0] + ',' + v[j][2] + ',' + v[j][1])
                            v[j][3] = True
        print(transaction_map)
        return invalid_transactions

# Medium (1/20/2024): https://leetcode.com/problems/two-city-scheduling/description/
class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        costs.sort(key=lambda person: person[0] - person[1])
        cur_sum = 0
        len_costs = len(costs)//2
        for i in range(len_costs):
            cur_sum += costs[i][0]
            cur_sum += costs[i+len_costs][1]

        return cur_sum


# Medium (1/20/2024): https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/
"""
# Definition for a Node.
class Node:
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child
"""

class Solution:
    def flatten(self, head: 'Optional[Node]') -> 'Optional[Node]':
        cur_p = head

        while cur_p:
            if cur_p.child:
                cur_tail = new_next = self.flatten(cur_p.child)
                while cur_tail.next:
                    cur_tail = cur_tail.next
                cur_next = cur_p.next
                cur_p.child= None
                cur_p.next = new_next
                new_next.prev = cur_p
                if cur_next:
                    cur_next.prev = cur_tail
                cur_tail.next = cur_next
                cur_p = cur_next

            else:
                cur_p = cur_p.next

        return head

# Medium (1/20/2024): https://leetcode.com/problems/design-a-leaderboard/
from collections import defaultdict

class Leaderboard:

    def __init__(self):
        self.leaderboard = defaultdict(int)
        

    def addScore(self, playerId: int, score: int) -> None:
        self.leaderboard[playerId] += score
        

    def top(self, K: int) -> int:
        heap_leader = []

        for p_score in self.leaderboard.values():
            if len(heap_leader) < K:
                heapq.heappush(heap_leader, p_score)
            else:
                if p_score > heap_leader[0]:
                    heapq.heappop(heap_leader)
                    heapq.heappush(heap_leader, p_score)

        return sum(heap_leader)

        return 0


        

    def reset(self, playerId: int) -> None:
        self.leaderboard[playerId] = 0
        


# Your Leaderboard object will be instantiated and called as such:
# obj = Leaderboard()
# obj.addScore(playerId,score)
# param_2 = obj.top(K)
# obj.reset(playerId)
        

# Medium (1/20/2024): https://leetcode.com/problems/design-underground-system/
        
from collections import defaultdict

class UndergroundSystem:
    def __init__(self):
        # Key = ID, Value = (stationName, time)
        self.people = {}
        # Key = (startStationName, endStationName), Value = (total time, num trips)
        self.visits = defaultdict(lambda : [0, 0])

    def checkIn(self, id: int, stationName: str, t: int) -> None:
        self.people[id] = (stationName, t)

    def checkOut(self, id: int, stationName: str, t: int) -> None:
        self.visits[(self.people[id][0], stationName)][0] += t - self.people[id][1]
        self.visits[(self.people[id][0], stationName)][1] += 1

    def getAverageTime(self, startStation: str, endStation: str) -> float:
        # Return the averge if it works, else return 0
        return self.visits[(startStation, endStation)][0]/self.visits[(startStation, endStation)][1] if self.visits[(startStation, endStation)][0] != 0 else 0
        


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)

# Medium (1/20/2024): https://leetcode.com/problems/design-browser-history/

class BrowserHistory:

    def __init__(self, homepage: str):
        self.history = [homepage]
        self.index = 0

    def visit(self, url: str) -> None:
        self.history = self.history[:(self.index+1)]
        self.history.append(url)
        self.index = len(self.history) -1
        
    def back(self, steps: int) -> str:
        self.index = max(self.index-steps, 0)
        return self.history[self.index]

    def forward(self, steps: int) -> str:
        self.index = min(self.index+steps, len(self.history)-1)
        return self.history[self.index]
        


# Your BrowserHistory object will be instantiated and called as such:
# obj = BrowserHistory(homepage)
# obj.visit(url)
# param_2 = obj.back(steps)
# param_3 = obj.forward(steps)


# Medium (1/20/2024): https://leetcode.com/problems/count-unhappy-friends/

class Solution:
    def unhappyFriends(self, n: int, preferences: List[List[int]], pairs: List[List[int]]) -> int:
        unhappy = set()
        pair_map = {}
        
        for p in pairs:
            pair_map[p[0]] = p[1]
            pair_map[p[1]] = p[0]
        
        for f in range(n):
            if f in unhappy:
                continue
            for pref_f in preferences[f]:
                # print(f, pref_f)
                if pair_map[f] == pref_f:
                    break
                if preferences[pref_f].index(pair_map[pref_f]) > preferences[pref_f].index(f):
                    unhappy.add(f)
                    unhappy.add(pref_f)
                    break
        return len(unhappy)
