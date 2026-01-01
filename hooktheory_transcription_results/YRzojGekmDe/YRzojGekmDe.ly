#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        g16*16 c16*16 g16*16 d16*6 c16*10 g16*16 c16*16 g16*16 d16*16 g16*16 c16*16 g16*16 d16*16 g16*16 c16*16 g16*16 d16*16:7
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 147
        r4. g'8 d''8 c''8 b'8 c''8 | c''1~ | c''4. b'4 a'8 g'8 a'8~ | a'8 a'2~ a'4.~ | a'4. d''4. b'8 d''8~ | d''1~ | d''2 b'4 g'8 a'8~ | a'16 a'2~ a'4~ a'8.~ | a'4 c''8 b'8 c''8 d''8 d''4 | d''8 c''8 c''8 c''4 c''4.~ | c''4 a'8 b'8 b'8 b'8 g'4 | a'8 a'2~ a'4.~ | a'4 d''8 d''8 d''8 d''8 d''8 d''8 | c''4 e''8 c''4 a'4.~ | a'4 b'2. | a'4. a'4 b'4 a'8
    }
}

>>

\version "2.18.2"