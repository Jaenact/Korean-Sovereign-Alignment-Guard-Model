from Moderation_API import guard_classify

while(1):
    question = input("프롬프트를 입력하세요.\n")
    res = guard_classify(question)
    print(res, "\n")