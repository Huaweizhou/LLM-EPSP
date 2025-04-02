################################################################未分类prompt
system_message = """You have ample time to carefully understand the varying degrees of influence of the following factors that determine whether a student will pass the final mathematics exam.

                    "age": "student's age (numeric: from 15 to 22)",
                    "studytime": "weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)",
                    "Medu": "mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",
                    "higher": "wants to take higher education (binary: yes or no)",
                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)",
                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)",
                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)",
                    "schoolsup": "extra educational support (binary: yes or no)".
                    Analyze the impact of various factors on students' ability to pass mathematics exams. 

                    Strictly based on the following principles, make predictions regarding students' likelihood of passing the exam:
                    1.Goout: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.
                    2.Higher: When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.
                    3.Age:
                    Students aged 21 or 22 will definitely fail the exam.
                    Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Students aged 18 or 20 are likely to pass the exam.
                    4.Failures:
                    Students with 0 failures are highly likely to pass the exam.
                    Students with 2 or 3 failures are highly likely to fail the exam.
                    Studytime: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.
                    5.Paid:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
                    Schoolsup:When schoolsup is 'no', it is very likely that the student will pass the exam.
                    6.Medu/Fedu:
                    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
                    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """

responses_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have ample time to think. Strictly adhere to the above principles, 

    make a judgment on whether the student {testcase} with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    1. Evaluate the Goout attribute.
    2. Evaluate the Higher attribute.
    3. Evaluate the Age attribute.
    4. Evaluate the Failures attribute.
    5. Evaluate the Studytime attribute.
    6. Evaluate the Paid attribute.
    7. Evaluate the Schoolsup attribute.
    8. Evaluate the Medu attributes and Fedu attributes.
    
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """
###############################################################################################分类的第一次尝试
system_message = """You have ample time to carefully understand the varying degrees of influence of the following factors that determine whether a student will pass the final mathematics exam.
                    Student's Social and Family Background:
                    "higher": "wants to take higher education (binary: yes or no)".
                    "Medu": "mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)".
                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)".
                    Student's Study Investment and Support:
                    "studytime": "weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)".
                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)".
                    "schoolsup": "extra educational support (binary: yes or no)".
                    Student's Behavior and Historical Performance:
                    "age": "student's age (numeric: from 15 to 22)".
                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)".
                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)".
                    
                    Analyze the impact of various factors on students' ability to pass mathematics exams. 

                    Strictly based on the following principles, make predictions regarding students' likelihood of passing the exam:
                    1.Student's Social and Family Background:
                    -Higher: When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.
                    -Medu/Fedu:
                    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
                    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
                    2.Student's Study Investment and Support:
                    -Studytime: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.
                    -Paid:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
                    -choolsup:When schoolsup is 'no', it is very likely that the student will pass the exam.
                    3.Student's Behavior and Historical Performance:
                    -Goout: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.
                    -Age:
                    Students aged 21 or 22 will definitely fail the exam.
                    Students aged 15, 16, or 17 are highly likely to pass the exam.
                    Students aged 18 or 20 are likely to pass the exam.
                    -Failures:
                    Students with 0 failures are highly likely to pass the exam.
                    Students with 2 or 3 failures are highly likely to fail the exam.
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """

responses_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have ample time to think. Strictly adhere to the above principles, 

    make a judgment on whether the student {testcase} with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    1.Evaluate the  Student's Social and Family Background
    -Evaluate the Higher attribute.
    -Evaluate the Medu attributes and Fedu attributes.
    2.Evaluate the  Student's Social and Family Background
    -Evaluate the Studytime attribute.
    -Evaluate the Paid attribute.
    -Evaluate the Schoolsup attribute.
    3.Evaluate the  Student's Social and Family Background
    -Evaluate the Goout attribute.
    -Evaluate the Age attribute. 
    -Evaluate the Failures attribute.
 
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """

###############################################################################################分类的第二次尝试
system_message = """You have ample time to carefully understand the varying degrees of influence of the following factors that determine whether a student will pass the final mathematics exam.
                    Student's Social and Family Background:
                    "higher": "wants to take higher education (binary: yes or no)".
                    "Medu": "mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)".
                    "Fedu": "father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)".
                    Student's Study Investment and Support:
                    "studytime": "weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)".
                    "paid": "extra paid classes within the course subject (Math) (binary: yes or no)".
                    "schoolsup": "extra educational support (binary: yes or no)".
                    Student's Behavior and Historical Performance:
                    "age": "student's age (numeric: from 15 to 22)".
                    "goout": "going out with friends (numeric: from 1 - very low to 5 - very high)".
                    "failures": "number of past class failures (numeric: n if 1<=n<3, else 4)".
                    
                    Analyze the impact of various factors on students' ability to pass mathematics exams. 
                    After thoroughly considering and fully understanding the impact of these factors, please confirm that you have understood.
                """

responses_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have ample time to think. Strictly adhere to the above principles, 

    make a judgment on whether the student {testcase} with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    Strictly based on the following principles, make predictions regarding students' likelihood of passing the exam:
    1.Evaluate the Student's Social and Family Background
    -Evaluate Higher attribute:When higher is 'yes', it is very likely that the student will pass the exam; when higher is 'no', it is very likely that the student will fail the exam.
    -Evaluate Medu/Fedu attribute:
    Students whose mother's education level (Medu) is 4 are very likely to pass the exam.
    Students whose father's education level (Fedu) is 4 are definitely to pass the exam.
    2.Evaluate the  Student's Social and Family Background
    -Evaluate the Studytime attribute: Students who study for 3 or 4 highly likely to pass the exam, but students studying for less than 3 are not guaranteed to pass.
    -Evaluate the Paid attribute:When Paid is 'yes', it is very likely that the student will pass the exam; when Paid is 'no', it is likely that the student will fail the exam.
    -Evaluate the Schoolsup attribute:When schoolsup is 'no', it is very likely that the student will pass the exam.
    3.Evaluate the  Student's Social and Family Background
    -Evaluate the Goout attribute: Students with goout values of 1, 2, or 3 are highly likely to pass the exam'.
    -Evaluate the Age attribute:
    Students aged 21 or 22 will definitely fail the exam.
    Students aged 15, 16, or 17 are highly likely to pass the exam.
    Students aged 18 or 20 are likely to pass the exam.
    -Evaluate the Failures attribute:
    Students with 0 failures are highly likely to pass the exam.
    Students with 2 or 3 failures are highly likely to fail the exam.
 
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """

#############################################################################################如何实现投票方法##############
responses_select_list = []
for testcase in x_test_instance:
    question_prompt = f"""You have ample time to think. Adhere to the above principles, 

    make a judgment on whether the student {testcase} with the following attributes can pass the math exam.(Pay special attention to degree words)

    If you believe the student can pass the math exam, answer 'yes'. If you believe the student cannot pass the math exam, answer 'no'. 
    
    Let's work this out in a step by step way to be sure we have the right answer.
    1. Evaluate the Goout attribute.
    2. Evaluate the Higher attribute.
    3. Evaluate the Age attribute.
    4. Evaluate the Failures attribute.
    5. Evaluate the Studytime attribute.
    6. Evaluate the Paid attribute.
    7. Evaluate the Schoolsup attribute.
    8. Evaluate the Medu attributes and Fedu attributes.
    
    Do not provide any other words and analysis.just 'yes' or 'no'.

    """
    responses_list = []
    for _ in range(5):
        response_1 = client.chat.completions.create(
            model=MODEL,
            messages = [{"role":"system","content":system_message}] + [{"role": "user","content":question_prompt}],
            stream = False,
            temperature=0.5,

        )
        response_test = response_1.choices[0].message.content
        responses_list.append(response_test)
        print(" response_test:", response_test)
        print("responses_list:",responses_list)
    response_2 = client.chat.completions.create(
            model=MODEL,
            messages = [{"role":"system","content":"You will be given 5 answers to the same question, please choose the answer that appears most often. There are only two possible answers, yes or no."}] + [{"role": "user","content":f"Please select the answer that appears most often from the 5 answers in {responses_list} and answer yes or no."}],
            stream = False,
            temperature=0,
        )
    response_select = response_2.choices[0].message.content
    responses_select_list.append(response_test)
    print("response_select:",response_select)
    print("responses_select_list:",responses_select_list)