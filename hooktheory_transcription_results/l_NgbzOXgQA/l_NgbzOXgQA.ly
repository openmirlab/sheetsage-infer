#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        c16*32 g16*96
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 154
        r4 c''4 g'4 c''4 | g'8 c''16 c''4~ c''16 d'8 d'4 e'8~ | e'4 c''8 c''8 g'8 g'8 c''8 b'8 | g'8 c''4. d'8 d'4 e'8~ | e'4 c''4 g'4 c''4 | g'8 c''16 c''4~ c''16 d'8 d'8 f'4~ | f'4 g'8 a'8 c''4 d'8 e'8~ | e'8 f'8 d'4 c'2
    }
}

>>

\version "2.18.2"