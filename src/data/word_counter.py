import argparse
import os

def count_words_line_by_line(file_path):
    total_words = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line == "<|endoftext|>\n":
                continue
            total_words += len(line.strip().split())
    return total_words

def count_words_by_story(file_path, delimiter="<|endoftext|>"):
    total_words = 0
    story_count = 0
    buffer = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if delimiter in line:
                buffer.append(line.replace(delimiter, ""))
                story_text = " ".join(buffer).strip()
                total_words += len(story_text.split())
                story_count += 1
                buffer = []
            else:
                buffer.append(line.strip())

        if buffer:
            story_text = " ".join(buffer).strip()
            total_words += len(story_text.split())
            story_count += 1

    return total_words, story_count

def get_all_files(path):
    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        for root, _, filenames in os.walk(path):
            for fname in filenames:
                full_path = os.path.join(root, fname)
                if os.path.isfile(full_path):
                    files.append(full_path)
    return files

def main():
    parser = argparse.ArgumentParser(description="Count total words in a file or folder (line-by-line or story-by-story).")
    parser.add_argument("-p", "--path", type=str, required=True,
                        help="Path to input file or folder")
    parser.add_argument("--mode", type=str, choices=["line", "story"], default="line",
                        help="Counting mode: 'line' for line-by-line, 'story' for <|endoftext|>-separated stories")
    parser.add_argument("--delimiter", type=str, default="<|endoftext|>",
                        help="Custom delimiter for story separation (used only in 'story' mode)")

    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Path not found: {args.path}")
        return

    all_files = get_all_files(args.path)
    if not all_files:
        print("No files found to process.")
        return

    total_words = 0
    total_stories = 0

    for file_path in all_files:
        try:
            if args.mode == "line":
                words = count_words_line_by_line(file_path)
                total_words += words
            else:
                words, stories = count_words_by_story(file_path, delimiter=args.delimiter)
                total_words += words
                total_stories += stories
        except Exception as e:
            print(f"Skipping {file_path}: {e}")

    print(f"\nScanned {len(all_files)} file(s).")
    if args.mode == "story":
        print(f"[STORY MODE] Total stories: {total_stories}")
    print(f"Total word count: {total_words:,}")

if __name__ == "__main__":
    main()
