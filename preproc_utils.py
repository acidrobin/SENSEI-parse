"""
preproc_utils.py

Provides utilities for loading and preprocessing datasets for training
or evaluation. Only includes get_combined_dataset and its dependencies.
"""

import pandas as pd
import datasets

# Mapping from dataset name to type
data_type_dict = {
    "sensei": "comment",
    "debatabase": "comment",
    "argessays": "essay",
    "argessays-oracle": "essay",
    "argessays-para": "essay",
    "argessays-para-oracle": "essay",
    "argessays-kawarada": "essay"
}

# Mapping from dataset name to CSV file path template
f_strings_dict = {
    "sensei": "sensei_data/sensei_end_to_end_{split}.csv",
    "debatabase": "debatabase_data/end_to_end_{split}_multilevel.csv",
    "argessays": "arg_annot_proc/end_to_end_{split}.csv",
    "argessays-oracle": "arg_annot_proc_oracle/end_to_end_{split}.csv",
    "argessays-para": "arg_annot_proc_paragraphs/end_to_end_{split}.csv",
    "argessays-para-oracle": "arg_annot_proc_paragraphs_oracle/end_to_end_{split}.csv",
    "argessays-kawarada": "arg_annot_proc_kawarada/end_to_end_{split}.csv"
}


def pandas2datasets(dataframe, seq2seq=False, tokenizer=None):
    """
    Convert a pandas DataFrame into a Hugging Face Dataset, optionally
    tokenized for seq2seq models.

    Args:
        dataframe: pd.DataFrame containing dataset.
        seq2seq: Boolean indicating if tokenizer should be applied.
        tokenizer: Pretrained tokenizer (required if seq2seq=True).

    Returns:
        datasets.Dataset
    """
    dataset = datasets.Dataset.from_pandas(dataframe)

    # Prompt templates
    full_text_comment = (
        f"<s>[INST]Create an Argument Graph from the comments below: \n{{comments}}\n---\nArgument Graph:[/INST]\n{{summaries}}[EOG]</s>"
    )
    prompt_comment = (
        f"<s>[INST]Create an Argument Graph from the comments below: \n{{comments}}\n---\nArgument Graph:[/INST]\n"
    )
    full_text_essay = (
        f"<s>[INST]Create an Argument Graph from the following essay: \n{{essays}}\n---\nArgument Graph:[/INST]\n{{summaries}}[EOG]</s>"
    )
    prompt_essay = (
        f"<s>[INST]Create an Argument Graph from the following essay: \n{{essays}}\n---\nArgument Graph:[/INST]\n"
    )

    def tokenize_add_label(prompt, summary):
        prompt_encoding = tokenizer.encode_plus(prompt)
        summary_encoding = tokenizer.encode_plus(summary)
        prompt_encoding["labels"] = summary_encoding["input_ids"]
        return prompt_encoding

    def apply_prompt_template(sample):
        """
        Apply appropriate prompt template and tokenize if needed.
        """
        if sample["essay_comment"] == "comment":
            if seq2seq:
                sample["summaries"] = sample["summaries"].replace("\n", "|")
                return tokenize_add_label(
                    prompt_comment.format(comments=sample["comments"]), sample["summaries"]
                )
            else:
                return {
                    "text": full_text_comment.format(comments=sample["comments"], summaries=sample["summaries"]),
                    "input": prompt_comment.format(comments=sample["comments"]),
                    "output": sample["summaries"]
                }
        else:  # essay
            if seq2seq:
                sample["summaries"] = sample["summaries"].replace("\n", "|")
                return tokenize_add_label(
                    prompt_essay.format(essays=sample["essays"]), sample["summaries"]
                )
            else:
                return {
                    "text": full_text_essay.format(essays=sample["essays"], summaries=sample["summaries"]),
                    "input": prompt_essay.format(essays=sample["essays"]),
                    "output": sample["summaries"]
                }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))
    return dataset


def get_combined_dataset(dsets_list, split, seq2seq=False, tokenizer=None):
    """
    Load and combine multiple datasets into a single Hugging Face Dataset.

    Args:
        dsets_list: List of dataset names (keys in f_strings_dict).
        split: "train" or "val".
        seq2seq: Boolean indicating if tokenizer should be applied.
        tokenizer: Pretrained tokenizer (required if seq2seq=True).

    Returns:
        datasets.Dataset
    """
    out_list = []

    for dset in dsets_list:
        # Handle argessays train/val slicing
        if "argessay" in dset:
            if split == "train":
                df = pd.read_csv(f_strings_dict[dset].format(split="train"))
                df["essay_comment"] = [data_type_dict[dset]] * len(df)
                out_list.append(df)
                df_val = pd.read_csv(f_strings_dict[dset].format(split="val"))
                df_val["essay_comment"] = [data_type_dict[dset]] * len(df_val)
                df_val = df_val[6:]
                out_list.append(df_val)
            elif split == "val":
                df_val = pd.read_csv(f_strings_dict[dset].format(split="val"))
                df_val["essay_comment"] = [data_type_dict[dset]] * len(df_val)
                df_val = df_val[:6]
                out_list.append(df_val)
        else:
            df = pd.read_csv(f_strings_dict[dset].format(split=split))
            df["essay_comment"] = [data_type_dict[dset]] * len(df)
            out_list.append(df)

    combined_df = pd.concat(out_list, ignore_index=True, sort=False)
    return pandas2datasets(combined_df, seq2seq=seq2seq, tokenizer=tokenizer)
