#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        c16*8:m ais16*8 dis16*8 gis16*8 c16*8:m ais16*8 gis16*16 c16*8:m ais16*8 dis16*8 gis16*8 dis16*8 f16*8:m c16*16:m
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 94
        r4 c''8 ais'8 g'8 f'8 dis'8. g'16~ | g'2 c'4. dis'8 | g'8 g'8 g'4 f'4 f'8 dis'8 | c'1~ | c'4 c''8 ais'8 g'8 f'8 dis'4~ | dis'4. c'16 c'4~ c'8. dis'8 | g'8 g'8 g'4 gis'4 gis'4~ | gis'2 g'2
    }
}

>>

\version "2.18.2"