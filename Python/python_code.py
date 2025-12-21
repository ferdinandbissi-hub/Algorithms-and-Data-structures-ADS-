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
