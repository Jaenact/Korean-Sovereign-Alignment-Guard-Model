from moderator_guard.classifier.core import guard_classify

while(1):
    question = input("Enter a prompt.\n")
    res = guard_classify(question)
    print(res, "\n")