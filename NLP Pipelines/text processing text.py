#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import textstat
import uuid
import time
import psycopg2
import datetime
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize
import cleantext
import contractions
import unicodedata
import pgenv


def remove_newline(text):
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")

    text = re.sub(r"([\s]{2,})", " ", text)

    return text


def remove_accented_chars(text):
    #     ```
    #     (NFKD) will apply the compatibility decomposition, i.e.
    #     replace all compatibility characters with their equivalents.
    #     ```
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def number_paragraphs(par):

    return list(
        zip([f"para://{str(par[0]).zfill(4)}" for par in list(enumerate(par))], par)
    )


def paragraph_tokenize(text):
    paragraphs = text.split("\n\n")

    numbered_paragraphs = number_paragraphs(paragraphs)

    return numbered_paragraphs


def clean_text(text):

    text = remove_newline(text)
    text = remove_accented_chars(text)

    cleantext.clean(
        text,
        fix_unicode=True,
        no_line_breaks=True,
        to_ascii=True,
        lower=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_digits=False,
        no_punct=False,
        no_emoji=True,
    )
    try:
        text = contractions.fix(text)
    except:
        pass

    return text


def number_paragraphs(par):

    return list(
        zip([f"para://{str(par[0]).zfill(4)}" for par in list(enumerate(par))], par)
    )


def paragraph_tokenize(text):
    paragraphs = text.split("\n\n")

    numbered_paragraphs = number_paragraphs(paragraphs)

    return numbered_paragraphs


conn_string = (
    fr"postgresql+psycopg2://postgres:{pgenv.password}@{pgenv.host}/medium_articles"
)
engine = create_engine(conn_string)

sql = """SELECT * FROM metadata"""

start_time = time.time()

print("date and time:", datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S"))
df = pd.read_sql(sql=sql, con=engine)

print(f"{datetime.timedelta(seconds=(time.time() - start_time))}")

start_time = time.time()
print("date and time:", datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S"))

df["sentences"] = df["text"].swifter.apply(sent_tokenize)

print("to_sql duration: {} seconds".format(time.time() - start_time))

df["paragraphs"] = df["text"].swifter.allow_dask_on_strings().apply(paragraph_tokenize)

df2 = df[["doc GUID", "paragraphs"]].explode("paragraphs")
df2 = df2.reset_index(drop=True)

df2[["para id", "paragraph"]] = pd.DataFrame(
    df2["paragraphs"].to_list(), columns=["para id", "paragraph"]
)
df2 = df2.drop(columns=["paragraphs"])
df = df.drop(columns=["paragraphs"])

# 12 minutes without swifter
# <6 minutes with swifter

start_time = time.time()
print("date and time:", datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S"))
df2["clean_paragraph"] = (
    df2["paragraph"].swifter.allow_dask_on_strings().apply(clean_text)
)
print(f"{datetime.timedelta(seconds=(time.time() - start_time))}")


start_time = time.time()
print("date and time:", datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S"))
df2["sentences"] = (
    df2["clean_paragraph"].swifter.allow_dask_on_strings().apply(sent_tokenize)
)
print(f"{datetime.timedelta(seconds=(time.time() - start_time))}")


df3 = df2[["doc GUID", "para id", "sentences"]].explode("sentences")
df3 = df3.reset_index(drop=True)
df2 = df2.drop(columns=["sent GUID", "sentences"])

df3["sent GUID"] = [f"sent://{str(uuid.uuid4())}" for x in range(len(df3))]
df3 = df3[["doc GUID", "para id", "sent GUID", "sentences"]]
df3 = df3[df3["sentences"].str.len() > 3]


# Runs in about 1 minute per million, for sentences

start_time = time.time()
print(
    "Running Flesch Reading Ease Score:",
    datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S"),
)
df3["flesch_reading_ease"] = (
    df3["sentences"].swifter.allow_dask_on_strings().apply(textstat.flesch_reading_ease)
)
print(f"{datetime.timedelta(seconds=(time.time() - start_time))}")


## Outputs the estimated gradelevel of reading. Yes, this will go over 12th grade, don't @ me

start_time = time.time()
print(
    "Running Readability Consensus:",
    datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S"),
)
df3["Readability Consensus"] = (
    df3["sentences"]
    .swifter.progress_bar(True)
    .allow_dask_on_strings()
    .apply(lambda x: textstat.text_standard(x, float_output=True))
)
print(f"{datetime.timedelta(seconds=(time.time() - start_time))}")


start_time = time.time()
print(
    "Counting difficult words:", datetime.datetime.now().strftime("%d/%b/%Y, %H:%M:%S")
)
df3["Difficult_words"] = (
    df3["sentences"]
    .swifter.progress_bar(True)
    .allow_dask_on_strings()
    .apply(textstat.difficult_words)
)
print(f"{datetime.timedelta(seconds=(time.time() - start_time))}")
