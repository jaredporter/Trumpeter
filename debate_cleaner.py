import argparse


def cleaner(filename,target,opponent,moderator):
    """
    Take a transcript of a debate and clean it up so that you only get
    the text from the person you want to use in your training data.

    Also, removes the applause from the transcript because why is it
    even there to begin with?
    """
    # Boolean to see if  text is from the target
    TARGET = True
    # Read the file
    with open(filename) as f:
        text = f.read()
    # Split into paragraphs
    nl_split = [x for x in text.split('\n') if x]
    # Empty list for target's speaking
    target_trans = []
    # Check who is speaking and if target, add it to the list
    for paragraph in nl_split:
        if paragraph.startswith(opponent):
            TARGET = False
        elif paragraph.startswith(moderator):
            TARGET = FALSE
        elif paragraph.startswith(target):
            TARGET = True

        if TARGET == True:
            target_trans.append(paragraph)

        # Join it into a single string
        target_trans = ' '.join(target_trans)

    applause = '(\(Applause\)|\(APPLAUSE\)|\(applause\)|\[Applause\]|\[APPLAUSE\]|\[applause\])'
    target_speaker = target.upper() + ': '
    target_trans = re.sub(applause, '', target_trans)
    target_trans = re.sub(target_speaker, '', target_trans)

    # Write it back to the file
    with open(filename, 'w') as f:
        f.write(target_trans)


if __name__ == '__main__':
    # Parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store', dest='filename')
    parser.add_argument('-t', action='store', dest='target')
    parser.add_argument('-o', action='store', dest='opponent')
    parser.add_argument('-m', action='store', dest='moderator')
    args = parser.parse_args()

    # Run the cleaner on the arguments from the parser
    cleaner(args.filename, args.target, args.opponent ,args.moderator)
