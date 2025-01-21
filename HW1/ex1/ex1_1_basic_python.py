# %% [markdown]
# # EX 1.1- basic python: Pyramid case
# Implement a function that get a string input and outputs the same word, only each odd char is lower
# case and each even letter is upper case
# You can assume that the input is a valid string which contains only english letters.
# %%


def pyramid_case(in_word):
    out_word = list(in_word)
    for i in range(len(in_word)):
        out_word[i] = in_word[i].lower() if i % 2 != 0 else in_word[i].upper()
    return ''.join(out_word)

# %%


def pyramid_case_one_liner(in_word):
    return ''.join(s_i.upper() if i % 2 == 0 else s_i.lower() for i, s_i in enumerate(in_word))


# %%
# test functions here
input_words = ["hello", "world", "", "I", "am", "LEARNING", "Python"]

print("==== pyramid_case() results:")
for word in input_words:
    print(pyramid_case(word))

print("\n==== pyramid_case_one_liner() results:")
for word in input_words:
    print(pyramid_case_one_liner(word))


# %%
