#(set-default-paper-size "letter")



<<

\new ChordNames {
    \set majorSevenSymbol = \markup { maj7 }
    \set additionalPitchPrefix = #"add"
    \chordmode {
        g16*24 gis16*24:maj7 g16*32 c16*32:m f16*24:m d16*8:dim ais16*8 f16*8 gis16*16 dis16*16:maj7 dis16*16:7 gis16*16 d16*48:7
    }
}

\new Staff {
    {
        \clef treble
        \key c \major
        \time 4/4
        \tempo 4 = 125
        g''1 | g'8 g'8 g'4 gis'2~ | gis'2. g'4 | g'2 gis'2~ | gis'2 b''2~ | b''1 | d''2. c''4~ | c''1~ | c''2. f''4 | f''2 f''4 f''4 | gis''1 | g''1~ | g''2~ g''4. dis''8~ | dis''1~ | dis''2~ dis''4. d''8~ | d''8 dis''2~ dis''4. | d'''4. c'''2 g''8
    }
}

>>

\version "2.18.2"