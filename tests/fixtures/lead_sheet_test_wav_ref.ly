#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        d16*32:7
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 170
        d''8 c''8 fis''8 e''4 d''4. | d''8 c''4 e''4. fis''4
    }
}

>>

\version "2.18.2"