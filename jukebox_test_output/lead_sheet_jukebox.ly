#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        d16*276
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 3/4
        \tempo 4 = 140
        r4 a'8 d'8 a'4 | e'2 a16 d'16 d'16 a'16 | a'8. e'16 e'2~ | e'2.~ | e'2.~ | e'2.~ | e'2.~ | e'2. | fis'8 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'8 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 | fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16 fis'16
    }
}

>>

\version "2.18.2"