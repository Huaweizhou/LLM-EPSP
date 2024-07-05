#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author zhouhuawei time:2024/3/4
import random

from langchain.output_parsers import ResponseSchema

response_schemas = [
    ResponseSchema(name="question", description="所出题目题干"),
    ResponseSchema(name="answer", description="所出题目答案"),
    ResponseSchema(name="knowledge", description="所出题目核心知识点"),
    ResponseSchema(name="testcase", description="所出题目测试示例演示(请给出5个演示示例)"),
    ResponseSchema(name="analyze", description="所出题目解析"),
    ResponseSchema(name="grade", description="所出题目的难度"),
    ResponseSchema(name="input_module", description="所出题目的答案的输入模板")
]

basic_requirement = """
    1.内容准确性：生成的问题应该在事实上是准确的，不应该包含任何误导性或错误的信息。同时，问题的答案也应该是准确和可验证的。
    2.清晰性：问题需要清晰明了，避免使用模糊或者含糊的表述。学生应该能够明确理解问题的要求。
    3.深度和复杂性：问题应该具有一定的深度和复杂性，以便测试学生的理解和应用能力，而不仅仅是记忆能力。
    4.实用性：问题应该尽可能地与实际生活或实际问题挂钩，以增加学生的学习兴趣和动机。
"""


def getSimilarRules() -> str:
    return """      1.知识内容与考点，确认两个题目是否基于同一知识点或者相同的概念、公式、定理等。
                    2.解题思路与方法，比较两个题目解题过程中所采用的逻辑推理路径和数学方法是否一致。
                    3.问题结构与形式,观察题目表述的形式和题干中的条件、问题设置是否存在模式上的相似性。
                    4.难度层次与考查目标，虽然题目表面看起来不同，但如果经过转换后实质上是在检验同一个数学能力或素养，那么也可以认为它们是相似题
            """


def getcode_1Example() -> str:
    example = """
 例子：questionStem:给你两个字符串 word1 和 word2 。请你从 word1 开始，通过交替添加字母来合并字符串。如果一个字符串比另一个字符串长，就将多出来的字母追加到合并后字符串的末尾。
        回答：
        {
            "question": "给定两个字符串word1和word2，将word1和word2按照首字母在字母表中的顺序，将二者进行合并，并输出合并后的字符串。",
            "knowledge": "字符串操作",
            "answer": "
        def merge_words(word1, word2):
            merged = ''
            i = 0  # 当前比较的字符索引

            # 按照首字母在字母表中的顺序合并字符串
            while i < len(word1) and i < len(word2):
                if word1[i] <= word2[i]:
                    merged += word1[i]
                    merged += word2[i]
                else:
                    merged += word2[i]
                    merged += word1[i]
               i += 1
    
             # 如果其中一个字符串还有剩余字符，直接将剩余字符添加到合并后的字符串中
            if i < len(word1):
                merged += word1[i:]
            if i < len(word2):
                merged += word2[i:]

            return merged "

            "testcase:":["
            word1 = "ace"
            word2 = "bdf"
            result = merge_words(word1, word2)
            print(result)  # 输出 "abcdef" ",
            "
            word1 = "aec"
            word2 = "bdf"
            result = merge_words(word1, word2)
            print(result)  # 输出 "abcdef"
            ",
            "
            word1 = "eca"
            word2 = "bdf"
            result = merge_words(word1, word2)
            print(result)  # 输出 "abcdef"
            ",
            "
            word1 = "cea"
            word2 = "bdf"
            result = merge_words(word1, word2)
            print(result)  # 输出 "abcdef"
            ",
            "
            word1 = "ace"
            word2 = "bdf"
            result = merge_words(word1, word2)
            print(result)  # 输出 "abcdef"
            "]
            
            "analyze":"我们定义了一个名为 merge_words 的函数，接受两个字符串 word1 和 word2 作为参数。该函数会按照首字母在字母表中的顺序合并两个字符串，并返回合并后的结果。
                       通过使用一个循环来比较两个字符串中对应位置的字符的大小关系，然后将较小的字符和较大的字符依次添加到合并后的字符串中，直到其中一个字符串的字符被遍历完。最后，如果其中一个字符串还有剩余字符，直接将剩余字符添加到合并后的字符串的末尾。
                       在示例中，我们使用了字符串 "ace" 和 "bdf" 进行合并，按照字母表中的顺序，合并后的结果为 "abcdef"。"     
            "grade":"1"
            "input_module":"def merge_words(word1, word2):
                            
                           "             
        }
                        """
    return example


def getcode_2Example() -> str:
    example = """
    例子:对Python列表中的元素进行移除操作可以使用remove()方法或者使用切片操作来删除特定位置的元素。
                回答：
                {
                    "question": "在Python编程中，给定一个包含重复元素的列表，编写一个函数去除列表中所有重复的元素，保持原有顺序不变？",
                    "knowledge": "Python列表中的元素进行移除操作",
                    "answer":"
                    def remove_duplicates(lst):
                        seen = set()
                        result = []
                        for item in lst:
                            if item not in seen:
                                seen.add(item)
                                result.append(item)
                        return result",
                    "case":["input: [1, 2, 3, 3, 4, 4, 5],
                            output: [1, 2, 3, 4, 5]",
                            "
                            input: [1, 2, 3, 3, 4, 4, 4, 5],
                            output: [1, 2, 3, 4, 5],
                            ",
                            "
                            input: [1, 2, 3, 3, 4, 4, 5, 5],
                            output: [1, 2, 3, 4, 5]
                            ",
                            "
                            input: [1, 2, 3, 3, 4, 4, 5, 5, 5],
                            output: [1, 2, 3, 4, 5]
                            ",
                            "
                            input: [1, 1, 1, 2, 3, 3, 4, 4, 5],
                            output: [1, 2, 3, 4, 5]
                            ",]
                    "analyze": "这个函数使用一个seen集合来跟踪已经出现过的元素。然后，它遍历列表中的每个元素，如果元素不在seen集合中，就将其添加到result列表中，并将其添加到seen集合中。这样就可以保证只有第一次出现的元素被添加到result列表中，而重复出现的元素会被忽略。"
                    "grade":"1"
                    "input_module":"def remove_duplicates(lst):"
                }
    """
    return example


def getcode_3Example() -> str:
    example = """
    例子：根据以下要求出一个高质量的题目。要求：题目考察python列表
    回答：{
    "question": "给定一个列表 nums 和一个目标值 target，请你在该列表中找出和为目标值的两个整数，并返回它们的索引。你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。你可以按任意顺序返回答案。",
    "answer": "def two_sum(nums, target):\n    hashmap = {}\n    for i, num in enumerate(nums):\n        complement = target - num\n        if complement in hashmap:\n            return [hashmap[complement], i]\n        hashmap[num] = i\n\nnums = [1, 3, 5, 7, 9]\ntarget = 10\nprint(two_sum(nums, target))",
    "knowledge": "Python列表、哈希表、枚举、查找算法",
    "case": [{"input": {"nums": [2, 7, 11, 15], "target": 9}, "output": [0, 1]}],
    "timeLimit": "100ms",
    "analyze": "这道题目考察了在给定列表中查找满足条件的两个数的索引。通过使用哈希表来存储已经遍历过的数值，可以将查找时间复杂度降低到 O(1)。遍历列表，计算当前数值与目标值的差值，若该差值已经存在于哈希表中，则返回该差值的索引以及当前数值的索引。如果没有找到满足条件的数对，则将当前数值加入哈希表中。"
    "grade":"1"
    "input_module":"def two_sum(nums, target):"
}
    """
    return example


def difficulty_standard(grade):
    difficulty_message = ""
    if grade == 0:
        difficulty_message = """
            本题的出题难度为简单，
            简单题目的出题规则可以从以下几个方面出发：
            （1）涉及的基础概念、公式或定义，要求学生简单回忆和陈述。
            （2）回答通常是一条或几条直接的定义、规则或事实，无需深入展开。
            （3）使用的案例或情境比较简单，直接对应某一知识点。
        """
    elif grade == 1:
        difficulty_message = """
            本题的出题难度为中等，
            简单题目的出题规则可以从以下几个方面出发：
            （1）要求考生不仅能回忆知识，还需进行适度的分析、比较或解释。
            （2）要求考生详细解释现象、过程或原理，并可能需要进行简要分析或举例说明。
            （3）案例或情境相对复杂，需要考生识别关键信息并运用所学知识。
        """
    elif grade == 2:
        difficulty_message = """
            本题的出题难度为复杂，
            简单题目的出题规则可以从以下几个方面出发：
            （1）要求考生深入理解概念背后的原理、机制或理论框架，能进行复杂的分析、推理或论证。 
            （2）要求考生进行严密的逻辑推理、批判性思考或创造性应用，可能还需要提供证据支持自己的观点。
            （3）案例或情境极具挑战性，可能涉及前沿研究、实践应用或复杂情境下的决策制定。
        """
    return difficulty_message

format_instructinos = """{
         "question": "所生成题目的题干",
         "knowledge": "题目考察的知识点",
         "analyze": "关于答案的解析",
         "answer": "所出题目答案，必须只回答选项对应的字母"
         "testcase":"所出题目测试示例演示,即输入什么，对应输出什么。需要给出5个输入输出的示例。"
         "grade":"请根据你的判断将难度等级分为1-5,并给出你认为此题的难度等级，回答为1，2，3，4，5中其中一个等级"
         "input_module":"给出解答解决此题的所需相关类Class()->,返回值return,函数定义def()流出空白代码可供人填写"
     }"""
