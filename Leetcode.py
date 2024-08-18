class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """

        if len(s) <= 1:
            return len(s)
        
        longest = 1
        start = s[0]
        keys = {}
        keys[s[0]] = 0
        back_index = 0
        front_index = 0
        for c in s[1:]:
            front_index += 1
            if c in start:
                # if len(start) > longest:
                if front_index - back_index > longest:
                    longest = front_index - back_index
                back_index = keys.get(c) + 1
                keys[c] = front_index
                start = c
            else:
                start += c
                keys[c] = front_index
        return max(longest, len(start))

test = Solution()
test.lengthOfLongestSubstring("vdvsd")
