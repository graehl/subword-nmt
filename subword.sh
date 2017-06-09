#!/bin/bash
d=`dirname $0`
set -e
tokenize() {
    if [[ $tokenizer ]] ; then
        $tokenizer
    else
        cat
    fi
}
vars() {
    echo "${sep:=__LW_SW__} ${subword_ops:=50000} ${subword_unk:=40} ${subword_prefix:=${subword_dir:-`dirname $0`}/subword} ${subword_outdir:=} ${exclude_bpe_basename:=1}" 1>&2
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
    ln -sf `absp $0` $subword_prefix
    codebase=$codes
    if [[ $joint ]] ; then
        python $d/learn_joint_bpe_and_vocab.py --input "$@" --separator $sep -s $subword_ops -o $codes --write-vocabulary $vocabs $versionarg
        apply "$@"
        for f in "$@"; do
            l=`lang "$f"`
            vocab=$(vocab $l)
            apply "$f"
        done
        ls -l $codes $vocabs
    else
        versionarg=$versionarg0
        for f in "$@"; do
            l=`lang "$f"`
            vocab=$(vocab $l)
            codes=$codebase.$l
            ls -l $code $vocab
            python $d/learn_bpe.py --input "$f" --separator $sep -s $subword_ops -o $codes --write-vocabulary $vocab $versionarg
            versionarg=$versionarg1
            apply "$f"
        done
    fi
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
            cat "$f" | lang=$l tokenizer | python -u $d/apply_bpe.py -s $sep -c $codes --vocabulary $vocab --vocabulary-threshold $subword_unk > $fto
        fi
    done
}
joint=1
        versionarg0=
        versionarg1=
case $1 in
    *01)
            joint=
            versionarg1=--version01
            ;;
    *10)
            joint=
            versionarg0=--version01
            ;;
    *00)
            joint=
            versionarg1=--version01
            versionarg0=--version01
            ;;
    *)
            ;;
esac
case $1 in
    -h*)
        echo 'usage: subword_prefix=/tmp/subword subword_ops=32000 subword_unk=50 $0 [-c] a.l1 b.l2'
        ;;
    -c*)
        subword_dir=${subword_dir:-.}
        vars
        shift
        create "$@"
        ;;
    *)
        vars
        apply "$@"
        ;;
esac
