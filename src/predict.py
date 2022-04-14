def compile_model_output(tokenizer, input_ids, model_output):
    ret = []

    for inp_id, label in zip(input_ids, model_output):
        token = tokenizer.convert_ids_to_tokens(inp_id.item())
        if token[:2] == "##":
            _tmp = ret[-1]
            ret[-1] = (tokenizer.convert_tokens_to_string([_tmp[0], token]), _tmp[1])
            continue

        ret.append((token, label))

    return ret


def removing_single_letter_abbreviations(compiled_output):
    for i, (word, label) in enumerate(compiled_output):
        if label[2:] == "short" and len(word) == 1:
            compiled_output[i] = (word, "O")


def remove_mislabeled_O_in_btwn_B_I(model_output):
    _tmp = []
    B_found_flag = False
    gap = 2

    for i, (word, label) in enumerate(model_output):
        if label == "O" and not B_found_flag:
            continue

        elif label[0] == "B":
            B_found_flag = True
            gap = 2
            _tmp = []
            continue

        elif label == "O":
            if gap > 0:
                _tmp.append(i)
                gap -= 1
            else:
                B_found_flag = True
                gap = 2
                _tmp = []
        elif label[0] == "I":
            if gap == 2:
                B_found_flag = False
            else:
                b_label = model_output[_tmp[0] - 1][1]
                if b_label[2:] == label[2:]:
                    for j in _tmp:
                        x, _ = model_output[j]
                        model_output[j] = (x, label)

                B_found_flag = True
                gap = 2
                _tmp = []


def get_long_short_pairs(model_output):
    long_form_found = False
    short_form_found = False

    long_form_start_end = [-1, -1]
    short_form_start_end = [-1, -1]

    gap = 3
    all_long_short_pairs = []

    for i, (word, label) in enumerate(model_output):
        if label == "O" and not long_form_found:
            continue
        elif label[2:] == "short" and not long_form_found:
            continue
        elif label[2:] == "long":
            if label[0] == "B":
                long_form_found = True
                long_form_start_end[0] = i
            else:
                long_form_start_end[1] = i
        elif label == "O" and not short_form_found:
            if gap > 0:
                gap -= 1
            else:
                long_form_found = False
                long_form_start_end = [-1, -1]
                gap = 3

        elif label[2:] == "short":
            if label[0] == "B":
                short_form_found = True
                short_form_start_end[0] = i
            else:
                short_form_start_end[1] = i

        elif label == "O" and long_form_found and short_form_found:

            # Long form str
            curr_long_form = []
            start, end = long_form_start_end
            if end == -1:
                curr_long_form = model_output[start][0]
            else:
                for j in range(start, end + 1):
                    curr_long_form.append(model_output[j][0])

                curr_long_form = " ".join(curr_long_form)

            # Short form str
            curr_short_form = []
            start, end = short_form_start_end
            if end == -1:
                curr_short_form = model_output[start][0]
            else:
                for j in range(start, end + 1):
                    curr_short_form.append(model_output[j][0])

                curr_short_form = " ".join(curr_short_form)

            # Add to the list
            all_long_short_pairs.append((curr_long_form, curr_short_form))

            # Reset
            long_form_found = False
            short_form_found = False

            long_form_start_end = [-1, -1]
            short_form_start_end = [-1, -1]

            gap = 3

    return all_long_short_pairs


def get_short_forms(model_output):
    ret = []
    for (word, label) in model_output:
        if label == "B-short":
            ret.append([word])
        elif label == "I-short":
            ret[-1].append(word)

    return ret


if __name__ == "__main__":
    test_input = [
        ("progenitor", "O"),
        ("virus", "O"),
        ("of", "O"),
        ("SARS", "B-short"),
        ("-", "I-short"),
        ("CoV", "I-short"),
        ("from", "O"),
        ("bats", "O"),
        ("have", "O"),
    ]
    print(get_short_forms(test_input))
