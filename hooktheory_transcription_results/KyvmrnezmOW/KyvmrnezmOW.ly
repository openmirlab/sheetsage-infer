#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        d16*32 a16*32:m e16*64:m d16*32 a16*32:m e16*64:m
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 194
        d''4 d''4 g''4 a''4~ | a''2. g'4 | a'4 b'4 d''4 g''4~ | g''2. a'8 b'8~ | b'1~ | b'1~ | b'1~ | b'1 | d''4 e''4 g''4 a''4~ | a''2. g'4 | a'4 b'4 b'4 g''4~ | g''4 e''2. | b'4 a'8 g'2~ g'8~ | g'1~ | g'1~ | g'1
    }
}

>>

\version "2.18.2"