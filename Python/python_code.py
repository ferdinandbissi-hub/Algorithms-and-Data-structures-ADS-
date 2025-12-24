# Day1: given an array of integer and integer target, return indices of two numbers such that they add up to target
import numpy as np
def return_indices(X,Y):
    n = len(X)
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] + arr[j] != target:
                continue
            else:
                return i,j

# Day 2: Given an integer array nums, find the subarray with the largest sum, and return its sum
def maxsum(nums):
    max_sum = 0
    max_sub_arr = []
    n = len(nums)
        
    for i in range(n):
        max_sub_arr.append(nums[i])
        new_max_sum = max_sum + nums[i]
        if new_max_sum > max_sum:
            max_sum = new_max_sum
        else:
            continue
    return new_max_sum

# Day 3: Given an array nums with n objects colored red, white, or blue, sort them inplace so that objects 
#of the same color are adjacent, with the colors in the order red, white and blue

# Let r,w, and b related to colours red, white and blue
def sort_colored_object(nums):
    n = len(nums)
    r,w,b = 0,0,0

    for i in nums:
        if i != 0 and i != 1 and i != 2:
            print("Your array should contain only 0,1 a" \
            "nd 2 as objects")
            break
    else:
        # Count the number of red, white and blue objects
        for i in range(n):
            if nums[i] == 0:
                r = r+1
            elif nums[i] == 1:
                w = w+1
            elif nums[i] == 2:
                b = b+1
            else:
                continue

        idx = 0
        for i in range(r):
            nums[idx] = 0
            idx = idx+1
        for i in range(w):
            nums[idx] = 1
            idx = idx+1
        for i in range(b):
            nums[idx] = 2
            idx = idx+1
    return nums
    
#Exercise 4: Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], 
#nums[c], nums[d]] such that: 0<=a, b,c,d<n a,b,c,and d are distinct nums[a]+nums[b]+nums[c]+nums[d] == target
def unique_qudruplet(nums, target):
    n = len(nums)
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    if nums[a]+nums[b]+nums[c]+nums[d] == target:
                        quadruplet = [nums[a],nums[b],nums[c],nums[d]]
                    else:
                        continue
    return quadruplet

# Day 5: Given an array of intervals where intervals[i] = [stati, endi], merge all overlapping intervals, and return 
  #and array of the non-overlapping intervals that cover all the intervals in the input
  def non_overlap(inter):
    inter = inter[np.argsort(inter[:,0])]  # Sort based on the start point of each interval

    print(inter)

    result = []
    for s,e in inter:
        if result and s <= result[-1][1]: # If result mean, does result has anything??
            result[-1][1] = max(result[-1][1], e)  # merge
        else:
            result.append([s,e])  # add
    return result

# Day 6: Given a string s of '(',')' and lowercase English characters.
#Your task is to remove the minimum number of parentheses ('('or')'in any positions) 
# that the resulting parentheses string is valid and return any valid sting

def remove_paren(s):
    list1 = []
    list2 = []
    count1 = 0
    count2 = 0

    for i in s:
        if i == "(":
            count1 += 1
            list1.append(i)
        elif i == ")":
            if count1 == 0:
                continue
            else:
                count1 -= 1
                list1.append(i)
        elif i != "(" and i != ")":
            list1.append(i)
    
    for i in reversed(list1):
        if i == ")":
            count2 += 1
            list2.append(i)
        elif i == "(":
            if count2 == 0:
                continue
            else:
                count2 -= 1
                list2.append(i)
        elif i != "(" and i != ")":
            list2.append(i)
    result = "".join(reversed(list2))
    return result

# Day 7: Given a string s, sort it in decreasing order based on the frequency of the characters. 
#The frequency of a character is the number of times it appears in the string.
#Return the sorted string. If there are multiple answers, return any of them

def sort_decreasing(s):
    dict = {} # store each character as a key and its frequency as the corresponding value

    for i in s:
        if i in dict:
            dict[i] += 1
        else:
            dict[i] = 1

    L = []
    for i in dict:
        L.append((dict[i],i)) 
    L.sort(key=lambda x: x[0], reverse=True)


    result = ""
    for i in range(len(L)):
        result = result + L[i][1]*L[i][0]
    return result


# Day 8: Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise. 
    #In the other words, return true if one of s1's permutations in the substring of s2
import itertools    # Python's Itertool is a module that provides various functions that work on iterators to produce complex iterators

def truefalse(s1, s2):
    perm = itertools.permutations(s1)
    result = [''.join(p) for p in perm] #str.join() is used to combine elements of an iterable (like a list, tuple, or set) into a single string, using a specified separator between each element
    print(result)
    result.remove(result[0])
    print(result)

    for i in result:
        if i in s2:
            answer = True
        else:
            answer = False
    return answer

# Day 9: Given a string s, partition s such that every substring of the partition is a Palindrome. 
    #Return all possible palindrome partitionning s

def palindrome(s):
    List1 = []

    for start in range(len(s)):
        for end in range(start+1, len(s)+1):
            substring = s[start:end]
            if substring == substring[::-1]:
                List1.append(substring)
    return List1

# Given two strings s and t of lengths m and n respectively, return the minimum window substring of s 
#such that every character in t (including duplicates) is included in the window. if ther is no such substring, return the empty string "".
def minimum_window(s,t):
    l1 = []
    l2 = []
    result = []
    if len(s) < len(t):
        return " "
    for k in t:
        l1.append(k)
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            l2.append(list(s[i:j]))

    for s in l2:
        if all(char in s for char in l1):
            if not result or len(s) < len(result):
                result = s

    if result:
        return "".join(result)
    else:
        return " "

# Day 11: Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val 
#and return the new head
def linked_list(L,val):
    for node in L[:]:
        if node == val:
            L.remove(node)
    return L

# Day 12: Given the head of a singly linked list, reverse the list and return the reversed list
def reverse_list(L):
    L.reverse()
    return L

# Day 13: Given an integer array nums of unique elements, return all possible subsets(the power set). 
#The solution set must not contain duplicate subsets. Return the solution in any order
import itertools
def combinations(nums):
    """ # In this code, l is the number of element in the combination.  For example
    for l = 0, the result is [].
    for  l = 1, we have one element per subset: [1], [2], [3].
    for l = 2, we have two elements per subset : [1, 2], [1, 3], [2, 3].
    for l = 3, we have three element in the subset : [1, 2, 3].
        """
    subset = []
    for l in range(len(nums)+1):
        for comb in itertools.combinations(nums, l):
            subset.append(list(comb))
    return subset

    
#Day 14: Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
def generate_parentheses(n):
    """
    Generate all combinations of well-formed parentheses for n pairs.
    
    Args:
    n (int): The number of pairs of parentheses.
    
    Returns:
    List[str]: A list of all combinations of well-formed parentheses.
    """
    result = []

    def generate(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        if open_count < n:
            generate(current + '(', open_count + 1, close_count)

        if close_count < open_count:
            generate(current + ')', open_count, close_count + 1)

    generate('', 0, 0)
    return result

#Day 16: Given an unsorted integer array nums, return the smallest missing positive integer. 
#You must implement an algorithm that runs in O(n) time and uses constant extra space
def first_missing_positive(nums):
    """
    This function finds the smallest missing positive integer from an unsorted integer array.
    
    It rearranges the array so that each positive integer x (if it is within the range 1 to n)
    is placed at the index x - 1. After rearranging, the function checks the array for the
    first index where the value does not match its expected value (index + 1).
    If all values are present, it returns n + 1, where n is the length of the array.
    """
    n = len(nums)
    
    # Rearranging the array
    for i in range(n):
        while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
            # Swap the elements to their correct positions
            nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
    
    # Finding the first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
            
    return n + 1

# Day 17: Given an m $\times$ n matrix, return all elements of the matrix in spiral order
def spiral_order(matrix):
    """
    Return all elements of the matrix in spiral order.
    """
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1  
    left, right = 0, len(matrix[0]) - 1 
    
    while top <= bottom and left <= right:
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        for col in range(right, left - 1, -1):
            result.append(matrix[bottom][col])
        bottom -= 1

        for row in range(bottom, top - 1, -1):
            result.append(matrix[row][left])
        left += 1
            
    return result

# Determine if a 9 $\times$ 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:
#Each row must contain the digits 1-9 without repetition.
#Each column must contain the digits 1-9 without repetition'
#Each of the nine 3 $\times$ 3 sub-boxes of the grid must contain the digits 1-9 without repetition

def is_valid_sudoku(board):
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    for i in range(9):
        for j in range(9):
            num = board[i][j]
            if num != '.':
                # Check row
                if num in rows[i]:
                    return False
                rows[i].add(num)

                # Check column
                if num in cols[j]:
                    return False
                cols[j].add(num)

                # Check box
                box_index = (i // 3) * 3 + (j // 3)
                if num in boxes[box_index]:
                    return False
                boxes[box_index].add(num)

    return True
