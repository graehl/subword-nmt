#!/bin/bash
realpath() {
    readlink -nfs $(cd "$(dirname $1)"; pwd)/"$(basename $1)"
}
d=`dirname $(realpath $0)`
set -e
tokenize() {
    if [[ $tokenizer ]] ; then
        $tokenizer
    else
        cat
    fi
}
bpevars() {
    echo "${sep:=__LW_SW__} ${subwords:=80000} ${mincount:=2} ${minfreq:=20} ${unkfreq:=2} ${subword_prefix:=${subword_dir:-`dirname $0`}/subword} ${subword_outdir:=} ${exclude_bpe_basename:=1} ${dict_input}" 1>&2
    codes=$subword_prefix.codes
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
    basename "$1" | awk -F "." '{print $2}'
}
create() {
    vocabs=""
    for f in "$@"; do
        l=`lang "$f"`
        vocabs+=" $(vocab $l)"
    done
    ln -sf `absp $0` $subword_prefix
    codebase=$codes
    if [[ $dict_input ]]  ; then
        dictinputarg=" --dict-input"
    fi
    if [[ $joint ]] ; then
        python $d/learn_joint_bpe_and_vocab.py --input "$@" --separator $sep -s $subwords -o $codes --write-vocabulary $vocabs $versionarg --min-frequency $minfreq --min-count $mincount $dictinputarg
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
            python $d/learn_bpe.py --input "$f" --separator $sep -s $subwords -o $codes --write-vocabulary $vocab $versionarg --min-frequency $minfreq --min-count $mincount
            versionarg=$versionarg1
            ls -l $code $vocab
            apply "$f"
        done
    fi
}
bpevocab() {
    vocab=$1
    shift
    python -u $d/apply_bpe.py -s $sep -c $codes --vocabulary $vocab --vocabulary-threshold $unkfreq "$@"
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
            (set -e
            [[ -s $vocab ]]
            cat "$f" | lang=$l tokenize | bpevocab $vocab > $fto
            )
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
        echo 'usage: subword_prefix=/tmp/subword subwords=32000 minfreq=50 $0 [-c] a.l1 b.l2'
        ;;
    -c*)
        if [[ ${1%v} != $1 ]] ; then
            dict_input=1
        fi
        subword_dir=${subword_dir:-.}
        bpevars
        shift
        create "$@"
        ;;
    *)
        bpevars
        apply "$@"
        ;;
esac
