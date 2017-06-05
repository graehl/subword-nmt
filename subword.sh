#!/bin/bash
set -e
tokenize() {
    if [[ $tokenizer ]] ; then
        $tokenizer
    else
        cat
    fi
}
vars() {
    echo "${subword_ops:=50000} ${subword_unk:=40} ${subword_prefix:=${subword_dir:-`dirname $0`}/subword} ${subword_outdir:=} ${exclude_bpe_basename:=1}" 1>&2
    codes=$subword_prefix.codes
    set -x
}
skipbpe() {
    [[ $exclude_bpe_basename ]] && basename "$1" | fgrep -q .bpe
}
absp() {
    readlink -nfs "$@"
}
vocab() {
    echo "${subword_prefix}.$1.vcb"
}
lang() {
    basename "$1" | awk -F "." '{print $NF}'
}
create() {
    vocabs=""
    for f in "$@"; do
        l=`lang "$f"`
        vocabs+=" $(vocab $l)"
    done
    learn_joint_bpe_and_vocab.py --input "$@" -s $subword_ops -o $codes --write-vocabulary $vocabs
    ln -sf `absp $0` $subword_prefix
    ls -l $codes $vocabs
    apply "$@"
}
apply() {
    for f in "$@"; do
        if ! skipbpe "$f" ; then
            l=`lang "$f"`
            fb=`basename -s .$l "$f"`
            vocab=`vocab $l`
            [[ $subword_outdir ]] || subword_outdir=`dirname "$f"`
            fto="$subword_outdir/$fb.bpe.$l"
            echo $fto
            lang=$l tokenizer "$f" | python -u apply_bpe.py -s __LW_SW__ -c $codes --vocabulary $vocab --vocabulary-threshold $subword_unk > $fto
        fi
    done
}
case $1 in
    -h*)
        echo 'usage: subword_prefix=/tmp/subword subword_ops=32000 subword_unk=50 $0 [-c] a.l1 b.l2'
        ;;
    -c*)
        shift
        subword_dir=${subword_dir:-.}
        vars
        create "$@"
        ;;
    *)
        vars
        apply "$@"
        ;;
esac
