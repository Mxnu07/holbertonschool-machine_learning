#!/usr/bin/env python3

def loop():
    """This function  takes in input from the user with the prompt Q:
    and prints A: as a response. If the user inputs exit, quit, goodbye,
    or bye, case insensitive, print A: Goodbye and exit. """

    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input('Q: ')
        if question.lower() in exit_commands:
            print('A: Goodbye')
            break
        print('A:')

if __name__ == '__main__':
    loop()
