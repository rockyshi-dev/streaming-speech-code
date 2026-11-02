"""Chapter 8: LocalAgreement-2, the commit policy of Whisper-Streaming, traced by hand.

    python code/ch08_whisper/la2_trace.py

A wrapper re-runs an offline model on a growing audio buffer and gets a new hypothesis each time.
LocalAgreement-2 commits a word only when two consecutive hypotheses agree on it: it commits the
longest common prefix of the last two hypotheses, after the words already committed. Words match as
exact strings, punctuation included, as in whisper_online.py (ufal/whisper_streaming, commit 6da90b4,
HypothesisBuffer.insert/flush, lines 371-417). This is a minimal re-implementation without the
timestamp and n-gram overlap handling; the hypotheses below are made up to show the policy.
No dependencies.
"""


def local_agreement_2(hypotheses):
    committed, previous, rows = [], None, []
    for k, hyp in enumerate(hypotheses, 1):
        new = hyp[len(committed):] if hyp[:len(committed)] == committed else hyp
        commit = []
        if previous is not None:
            for a, b in zip(previous, new):
                if a != b:
                    break
                commit.append(a)
        committed += commit
        previous = new[len(commit):]
        rows.append((k, " ".join(hyp), " ".join(commit) or "-", " ".join(previous) or "-"))
    return committed, rows


if __name__ == "__main__":
    # one hypothesis per update, after 1, 2, 3 and 4 s of audio
    hyps = [["Thank", "you,", "Mr."],
            ["Thank", "you,", "Mr.", "President."],
            ["Thank", "you,", "Mr.", "President.", "Today"],
            ["Thank", "you,", "Mr.", "President.", "Today,", "I"]]
    committed, rows = local_agreement_2(hyps)
    print(f"{'update':<7}{'hypothesis':<42}{'committed now':<20}pending")
    for k, h, c, p in rows:
        print(f"{k:<7}{h:<42}{c:<20}{p}")
    print("committed after 4 updates:", " ".join(committed))
