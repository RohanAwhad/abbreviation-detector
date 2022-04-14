def compile_model_output(tokenizer, batch_input_ids, batch_model_output, OVERLAP):
    ret = []

    for input_ids, model_output in zip(batch_input_ids, batch_model_output):
        _ignore = []
        for inp_id, label in zip(input_ids, model_output):
            token = tokenizer.convert_ids_to_tokens(inp_id.item())
            if token == "[CLS]":
                continue
            if token == "[SEP]":
                break
            if token[:2] == "##":
                _tmp = _ignore[-1]
                _ignore[-1] = (
                    tokenizer.convert_tokens_to_string([_tmp[0], token]),
                    _tmp[1],
                )
                continue

            _ignore.append((token, label))

        """
        if len(ret) > 0:
            _tmp_overlap = OVERLAP
            overlapping_part_1 = ret[-_tmp_overlap:]
            overlapping_part_2 = _ignore[:_tmp_overlap]

            if len(overlapping_part_2) < OVERLAP:
                _tmp_overlap = len(overlapping_part_2)
                overlapping_part_1 = ret[-_tmp_overlap:]

            # Sync last and first of the overlapping part
            if (
                overlapping_part_1[-1] != overlapping_part_2[-1]
                and overlapping_part_1[-1] == _ignore[_tmp_overlap]
            ):
                overlapping_part_2 = _ignore[: _tmp_overlap + 1]

            if (
                overlapping_part_2[0] != overlapping_part_1[0]
                and overlapping_part_2[0] == ret[-(_tmp_overlap + 1)]
            ):
                overlapping_part_1 = ret[-(_tmp_overlap + 1) :]

            print(overlapping_part_1)
            print(overlapping_part_2)
            if len(overlapping_part_1) != _tmp_overlap:
                _tmp_overlap = len(overlapping_part_1)

            assert len(overlapping_part_1) == len(overlapping_part_2)

            resolved_list = []
            # Give proper labelling of the overlapping part
            for (token_A, label_A), (token_B, label_B) in zip(
                overlapping_part_1, overlapping_part_2
            ):
                assert token_A == token_B

                if label_A == label_B:
                    resolved_list.append((token_A, label_A))
                    continue
                else:
                    if label_A == "O":
                        resolved_list.append((token_B, label_B))
                    elif label_B == "O":
                        resolved_list.append((token_A, label_A))
                    else:
                        raise Exception(
                            f"For token: '{token_A}' 2 predictions are \
                                given: {label_A} and {label_B}"
                        )

            ret = ret[:-_tmp_overlap] + resolved_list + _ignore[_tmp_overlap:]
            # ret.extend(_ignore)
        else:
            ret.extend(_ignore)
        """ 
        ret.extend(_ignore)

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

    if long_form_found and short_form_found:
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
        ("the", "O"),
        ("human", "O"),
        ("angiotensin", "B-long"),
        ("converting", "I-long"),
        ("enzyme", "I-long"),
        ("II", "I-long"),
        ("(", "O"),
        ("ACE2", "B-short"),
    ]
    print(get_long_short_pairs(test_input))
