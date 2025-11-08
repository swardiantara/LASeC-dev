import win32com.client as win32
import os

merge_review = """
Sub HighlightOverlaps()
    Dim ws As Worksheet
    Dim lastRow As Long
    Dim headerRow As Long
    Dim colA As Long, colB As Long
    Dim i As Long
    Dim textA As String, textB As String
    Dim tokensA() As String, tokensB() As String
    Dim overlapList As Object
    Dim overlap As Variant
    Dim cellA As Range, cellB As Range
    Dim colorPalette As Variant
    Dim colorIndex As Long

    ' ==== CONFIGURATION ====
    headerRow = 1          ' Row where your column names appear
    Set ws = ThisWorkbook.Sheets("Sheet1")

    ' ==== Identify columns by name ====
    colA = WorksheetFunction.Match("first_prototype", ws.Rows(headerRow), 0)
    colB = WorksheetFunction.Match("second_prototype", ws.Rows(headerRow), 0)

    ' ==== Define color palette ====
    colorPalette = Array(RGB(220, 20, 60), RGB(25, 25, 112), RGB(0, 100, 0), RGB(138, 43, 226), RGB(0, 139, 139), RGB(199, 21, 133), RGB(210, 105, 30), RGB(70, 130, 180), RGB(178, 34, 34), RGB(47, 79, 79))


    lastRow = ws.Cells(ws.Rows.Count, colA).End(xlUp).Row

    ' ==== Process each row ====
    For i = headerRow + 1 To lastRow
        Set cellA = ws.Cells(i, colA)
        Set cellB = ws.Cells(i, colB)

        textA = Trim(cellA.Value & "")
        textB = Trim(cellB.Value & "")

        ' Reset formatting
        cellA.Font.ColorIndex = xlAutomatic
        cellB.Font.ColorIndex = xlAutomatic

        If Len(textA) > 0 And Len(textB) > 0 Then
            tokensA = Split(LCase(textA))
            tokensB = Split(LCase(textB))

            Set overlapList = CreateObject("Scripting.Dictionary")

            ' ---- Find all overlapping n-grams ----
            Dim aStart As Long, bStart As Long, aLen As Long
            For aStart = 0 To UBound(tokensA)
                For bStart = 0 To UBound(tokensB)
                    aLen = 0
                    Do While (aStart + aLen) <= UBound(tokensA) And (bStart + aLen) <= UBound(tokensB)
                        If tokensA(aStart + aLen) = tokensB(bStart + aLen) Then
                            aLen = aLen + 1
                            If aLen > 0 Then
                                Dim seq As String
                                seq = Join(SubArray(tokensA, aStart, aStart + aLen - 1), " ")
                                If Not overlapList.exists(seq) Then overlapList.Add seq, aLen
                            End If
                        Else
                            Exit Do
                        End If
                    Loop
                Next bStart
            Next aStart

            ' ---- Filter out overlaps contained in longer overlaps ----
            Dim filteredList As Object
            Set filteredList = CreateObject("Scripting.Dictionary")
            Dim longer As Variant, shorter As Variant
            Dim keep As Boolean

            For Each shorter In overlapList.Keys
                keep = True
                For Each longer In overlapList.Keys
                    If Len(longer) > Len(shorter) Then
                        If InStr(1, longer, shorter, vbTextCompare) > 0 Then
                            keep = False
                            Exit For
                        End If
                    End If
                Next longer
                If keep Then filteredList(shorter) = overlapList(shorter)
            Next shorter
            Set overlapList = filteredList


            ' ---- Highlight overlaps ----
            colorIndex = 0
            For Each overlap In overlapList.Keys
                Dim thisColor As Long
                thisColor = colorPalette(colorIndex Mod (UBound(colorPalette) + 1))
                Call HighlightText(cellA, overlap, thisColor)
                Call HighlightText(cellB, overlap, thisColor)
                colorIndex = colorIndex + 1
            Next overlap
        End If
    Next i

    MsgBox "Highlighting complete for Sheet1!", vbInformation
End Sub

'--------------------------------------------------------------
' Helper: safely extract a portion of a string array
'--------------------------------------------------------------
Private Function SubArray(arr() As String, startIdx As Long, endIdx As Long) As Variant
    Dim temp() As String
    Dim i As Long, j As Long
    If endIdx < startIdx Then
        ReDim temp(0 To 0)
        temp(0) = ""
        SubArray = temp
        Exit Function
    End If
    ReDim temp(0 To endIdx - startIdx)
    j = 0
    For i = startIdx To endIdx
        temp(j) = arr(i)
        j = j + 1
    Next i
    SubArray = temp
End Function

'--------------------------------------------------------------
' Helper: highlight substring (case-insensitive)
'--------------------------------------------------------------
Private Sub HighlightText(cell As Range, ByVal phrase As String, ByVal color As Long)
    Dim text As String
    Dim pos As Long

    If Len(Trim(phrase)) = 0 Then Exit Sub
    text = LCase(cell.Value & "")
    phrase = LCase(phrase)
    pos = InStr(1, text, phrase)

    Do While pos > 0
        cell.Characters(pos, Len(phrase)).Font.Color = color
        pos = InStr(pos + Len(phrase), text, phrase)
    Loop
End Sub


Sub AddVerdictDropdown()
    Dim ws As Worksheet
    Dim headerRow As Long
    Dim colFirst As Long, colSecond As Long, colVerdict As Long
    Dim lastRow As Long
    Dim i As Long
    Dim rng As Range
    
    Set ws = ThisWorkbook.Sheets("Sheet1")   ' adjust if needed
    headerRow = 1                             ' header row
    
    On Error Resume Next
    colFirst = WorksheetFunction.Match("first_prototype", ws.Rows(headerRow), 0)
    colSecond = WorksheetFunction.Match("second_prototype", ws.Rows(headerRow), 0)
    colVerdict = WorksheetFunction.Match("verdict", ws.Rows(headerRow), 0)
    On Error GoTo 0
    
    If colFirst = 0 Or colSecond = 0 Or colVerdict = 0 Then
        MsgBox "Missing one or more required headers: 'first_prototype', 'second_prototype', or 'verdict'.", vbExclamation
        Exit Sub
    End If
    
    ' --- Find last non-empty row across the two prototype columns ---
    lastRow = headerRow
    For i = ws.Rows.Count To headerRow + 1 Step -1
        If Trim(CStr(ws.Cells(i, colFirst).Value)) <> "" Or _
           Trim(CStr(ws.Cells(i, colSecond).Value)) <> "" Then
            lastRow = i
            Exit For
        End If
    Next i
    
    If lastRow <= headerRow Then
        MsgBox "No valid data found in prototype columns.", vbInformation
        Exit Sub
    End If
    
    ' --- Apply dropdown + default value only to rows with data ---
    For i = headerRow + 1 To lastRow
        If Trim(CStr(ws.Cells(i, colFirst).Value)) <> "" Or _
           Trim(CStr(ws.Cells(i, colSecond).Value)) <> "" Then
           
            Set rng = ws.Cells(i, colVerdict)
            
            ' Add dropdown (change keep,merge to keep,split if needed)
            With rng.Validation
                .Delete
                .Add Type:=xlValidateList, AlertStyle:=xlValidAlertStop, _
                     Operator:=xlBetween, Formula1:="keep,merge"
                .IgnoreBlank = True
                .InCellDropdown = True
                .ShowError = True
            End With
            
            ' Set default if empty
            If Trim(CStr(rng.Value)) = "" Then rng.Value = "keep"
        End If
    Next i
    
    MsgBox "Dropdowns successfully added for rows with prototype data.", vbInformation
End Sub
"""

split_review = """
Sub HighlightOverlaps()
    Dim ws As Worksheet
    Dim lastRow As Long
    Dim headerRow As Long
    Dim colA As Long, colB As Long
    Dim i As Long
    Dim textA As String, textB As String
    Dim tokensA() As String, tokensB() As String
    Dim overlapList As Object
    Dim overlap As Variant
    Dim cellA As Range, cellB As Range
    Dim colorPalette As Variant
    Dim colorIndex As Long

    ' ==== CONFIGURATION ====
    headerRow = 1          ' Row where your column names appear
    Set ws = ThisWorkbook.Sheets("Sheet1")

    ' ==== Identify columns by name ====
    colA = WorksheetFunction.Match("prototype_message", ws.Rows(headerRow), 0)
    colB = WorksheetFunction.Match("member_message", ws.Rows(headerRow), 0)

    ' ==== Define color palette ====
    colorPalette = Array(RGB(220, 20, 60), RGB(25, 25, 112), RGB(0, 100, 0), RGB(138, 43, 226), RGB(0, 139, 139), RGB(199, 21, 133), RGB(210, 105, 30), RGB(70, 130, 180), RGB(178, 34, 34), RGB(47, 79, 79))


    lastRow = ws.Cells(ws.Rows.Count, colA).End(xlUp).Row

    ' ==== Process each row ====
    For i = headerRow + 1 To lastRow
        Set cellA = ws.Cells(i, colA)
        Set cellB = ws.Cells(i, colB)

        textA = Trim(cellA.Value & "")
        textB = Trim(cellB.Value & "")

        ' Reset formatting
        cellA.Font.ColorIndex = xlAutomatic
        cellB.Font.ColorIndex = xlAutomatic

        If Len(textA) > 0 And Len(textB) > 0 Then
            tokensA = Split(LCase(textA))
            tokensB = Split(LCase(textB))

            Set overlapList = CreateObject("Scripting.Dictionary")

            ' ---- Find all overlapping n-grams ----
            Dim aStart As Long, bStart As Long, aLen As Long
            For aStart = 0 To UBound(tokensA)
                For bStart = 0 To UBound(tokensB)
                    aLen = 0
                    Do While (aStart + aLen) <= UBound(tokensA) And (bStart + aLen) <= UBound(tokensB)
                        If tokensA(aStart + aLen) = tokensB(bStart + aLen) Then
                            aLen = aLen + 1
                            If aLen > 0 Then
                                Dim seq As String
                                seq = Join(SubArray(tokensA, aStart, aStart + aLen - 1), " ")
                                If Not overlapList.exists(seq) Then overlapList.Add seq, aLen
                            End If
                        Else
                            Exit Do
                        End If
                    Loop
                Next bStart
            Next aStart

            ' ---- Filter out overlaps contained in longer overlaps ----
            Dim filteredList As Object
            Set filteredList = CreateObject("Scripting.Dictionary")
            Dim longer As Variant, shorter As Variant
            Dim keep As Boolean

            For Each shorter In overlapList.Keys
                keep = True
                For Each longer In overlapList.Keys
                    If Len(longer) > Len(shorter) Then
                        If InStr(1, longer, shorter, vbTextCompare) > 0 Then
                            keep = False
                            Exit For
                        End If
                    End If
                Next longer
                If keep Then filteredList(shorter) = overlapList(shorter)
            Next shorter
            Set overlapList = filteredList


            ' ---- Highlight overlaps ----
            colorIndex = 0
            For Each overlap In overlapList.Keys
                Dim thisColor As Long
                thisColor = colorPalette(colorIndex Mod (UBound(colorPalette) + 1))
                Call HighlightText(cellA, overlap, thisColor)
                Call HighlightText(cellB, overlap, thisColor)
                colorIndex = colorIndex + 1
            Next overlap
        End If
    Next i

    MsgBox "Highlighting complete for Sheet1!", vbInformation
End Sub

'--------------------------------------------------------------
' Helper: safely extract a portion of a string array
'--------------------------------------------------------------
Private Function SubArray(arr() As String, startIdx As Long, endIdx As Long) As Variant
    Dim temp() As String
    Dim i As Long, j As Long
    If endIdx < startIdx Then
        ReDim temp(0 To 0)
        temp(0) = ""
        SubArray = temp
        Exit Function
    End If
    ReDim temp(0 To endIdx - startIdx)
    j = 0
    For i = startIdx To endIdx
        temp(j) = arr(i)
        j = j + 1
    Next i
    SubArray = temp
End Function

'--------------------------------------------------------------
' Helper: highlight substring (case-insensitive)
'--------------------------------------------------------------
Private Sub HighlightText(cell As Range, ByVal phrase As String, ByVal color As Long)
    Dim text As String
    Dim pos As Long

    If Len(Trim(phrase)) = 0 Then Exit Sub
    text = LCase(cell.Value & "")
    phrase = LCase(phrase)
    pos = InStr(1, text, phrase)

    Do While pos > 0
        cell.Characters(pos, Len(phrase)).Font.Color = color
        pos = InStr(pos + Len(phrase), text, phrase)
    Loop
End Sub


Sub AddVerdictDropdown()
    Dim ws As Worksheet
    Dim headerRow As Long
    Dim colFirst As Long, colSecond As Long, colVerdict As Long
    Dim lastRow As Long
    Dim i As Long
    Dim rng As Range
    
    Set ws = ThisWorkbook.Sheets("Sheet1")   ' adjust if needed
    headerRow = 1                             ' header row
    
    On Error Resume Next
    colFirst = WorksheetFunction.Match("prototype_message", ws.Rows(headerRow), 0)
    colSecond = WorksheetFunction.Match("member_message", ws.Rows(headerRow), 0)
    colVerdict = WorksheetFunction.Match("verdict", ws.Rows(headerRow), 0)
    On Error GoTo 0
    
    If colFirst = 0 Or colSecond = 0 Or colVerdict = 0 Then
        MsgBox "Missing one or more required headers: 'prototype_message', 'member_message', or 'verdict'.", vbExclamation
        Exit Sub
    End If
    
    ' --- Find last non-empty row across the two prototype columns ---
    lastRow = headerRow
    For i = ws.Rows.Count To headerRow + 1 Step -1
        If Trim(CStr(ws.Cells(i, colFirst).Value)) <> "" Or _
           Trim(CStr(ws.Cells(i, colSecond).Value)) <> "" Then
            lastRow = i
            Exit For
        End If
    Next i
    
    If lastRow <= headerRow Then
        MsgBox "No valid data found in prototype columns.", vbInformation
        Exit Sub
    End If
    
    ' --- Apply dropdown + default value only to rows with data ---
    For i = headerRow + 1 To lastRow
        If Trim(CStr(ws.Cells(i, colFirst).Value)) <> "" Or _
           Trim(CStr(ws.Cells(i, colSecond).Value)) <> "" Then
           
            Set rng = ws.Cells(i, colVerdict)
            
            ' Add dropdown (change keep,merge to keep,split if needed)
            With rng.Validation
                .Delete
                .Add Type:=xlValidateList, AlertStyle:=xlValidAlertStop, _
                     Operator:=xlBetween, Formula1:="keep,split"
                .IgnoreBlank = True
                .InCellDropdown = True
                .ShowError = True
            End With
            
            ' Set default if empty
            If Trim(CStr(rng.Value)) = "" Then rng.Value = "keep"
        End If
    Next i
    
    MsgBox "Dropdowns successfully added for rows with prototype data.", vbInformation
End Sub
"""


def apply_macro(excel_path, macro_code):
    # Launch Excel
    excel = win32.Dispatch("Excel.Application")
    excel.Visible = False  # set True to watch Excel running
    
    # Open workbook
    wb = excel.Workbooks.Open(excel_path)

    # Add macro temporarily
    vb_module = wb.VBProject.VBComponents.Add(1)  # 1 = standard module
    vb_module.CodeModule.AddFromString(macro_code)

    # Run macro
    excel.Run("HighlightOverlaps")
    excel.Run("AddVerdictDropdown")

    # Save & close
    wb.Save()
    wb.Close(SaveChanges=True)

    excel.Quit()

file_a = os.path.join('review_sheets_raw', 'merge_review.annotator_A.xlsx')
file_b = os.path.join('review_sheets_raw', 'merge_review.annotator_B.xlsx')
file_c = os.path.join('review_sheets_raw', 'split_review.annotator_A.xlsx')
file_d = os.path.join('review_sheets_raw', 'split_review.annotator_B.xlsx')

apply_macro(os.path.abspath(file_a), merge_review)
apply_macro(os.path.abspath(file_b), merge_review)
apply_macro(os.path.abspath(file_c), split_review)
apply_macro(os.path.abspath(file_d), split_review)