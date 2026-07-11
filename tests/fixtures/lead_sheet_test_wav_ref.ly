#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        g16*36
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 3/4
        \tempo 4 = 120
        r4. b'8. d''8.~ | d''4 d''4 d''4~ | d''2 g'4
    }
}

>>

\version "2.18.2"