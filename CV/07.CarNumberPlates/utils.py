import cv2
import numpy as np
import re

# functions to transform perspective of license plate
def compute_output_size(pts):
    """
    Compute dynamic size of license plate.

    Args:
        pts (array): 4 points [top-left, top-right, bottom-right, bottom-left]
    Returns:
        maxWidth (int): Maximum Width
        maxHeight (int): Maximum Height
    """
    pts = [np.array(p) for p in pts]
    (tl, tr, br, bl) = pts
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    maxHeight = int(max(heightA, heightB))

    return maxWidth, maxHeight

def warp_perspective(frame, pts):
    """
    Format the license plate perspective.


    Args:
        frame(numpy.ndarray): Frame with the license plate
        pts(array): 4 points [top-left, top-right, bottom-right, bottom-left]
    Returns:
        warped(array): Formatted license plate.
    """
    width, height = compute_output_size(pts)
    dst = np.array([[0,0],[width-1,0], [width-1, height-1], [0, height-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(np.array(pts, dtype='float32'), dst)
    warped = cv2.warpPerspective(frame, M, (width, height))
    return warped

# functions to change the number after easyOCR detect numbers and chars
"""
russian license plate is:
Л111ОХ01 or Л111ОХ001
len can be in [8,9] with region number or in [6, 7] without region numbed
region number is two or three last chars in number
"""

allowed_symbols = 'АВЕКМНОРСТУХ0123456789'
allowed_chars = 'АВЕКМНОРСТУХ'
prohibited_chars = 'ЙЦГШЩЗФЫПЛДЖЭЯЧЬЪБЮ'

wrong_symbols = {'#':'Н',
               ']':'1',
               '[':'1',
               '(':'1',
               ')':'1',
               '{':'1',
               '}':'1',
               '!':'1',
               'Ч':'4',
               'Ф':'О',
               'С':'О',
               'З':'3',
               'Ь':'6',
               'Б':'6',
               'Ъ':'6',
               'Ю':'О'}

dict_char_to_num = {'O':'0',
                    'А':'4',
                    'В':'8',
                    'Т':'7'}

dict_num_to_char = {'0':'О',
                    '6':'О',
                    '9':'О',
                    '4':'А',
                    '8':'В',
                    '7':'Т'}

def correct_common_ocr_errors(text):
    """
    Change wrong synbols on correct chars or numbers

    Args:
        text(str): easyOCR predicted text
    Returns:
        text(str): text with changed symbols
    """
    for wrong, correct in wrong_symbols.items():
        text = text.replace(wrong, correct)
    if text[0] == '1': # license can't start from 1, only chars=)
        text = text[1:]
    return text

def clean_plate(text):
    """
    Сlears text from other unwanted characters

    Args:
        text(str): easyOCR predicted text
    Returns:
        text(str): text with clean symbols
    """
    
    cleaned_text = ''.join(char for char in text if char in allowed_symbols)
    if len(cleaned_text) > 5:
        return cleaned_text
    else:
        return None
    

# detect number and region in license plate
def correct_number(text):
    if len(text) < 5:
        return False
    # because i work in first 6 symbols in this project, i drop other in this func
    text = text[:6]

    if (text[0] in allowed_chars or text[0] in dict_num_to_char.keys()) and \
       (text[1].isdigit() or text[1] in dict_char_to_num.keys()) and \
       (text[2].isdigit() or text[2] in dict_char_to_num.keys()) and \
       (text[3].isdigit() or text[3] in dict_char_to_num.keys()) and \
       (text[4] in allowed_chars or text[4] in dict_num_to_char.keys()) and \
       (text[5] in allowed_chars or text[5] in dict_num_to_char.keys()):
        return True
    else:
        return False
    
def format_license(text):
    """
    Format the license plate text by converting characters using the mapping dictionaries.

    Args:
        text (str): License plate text.

    Returns:
        text(str): Formatted license plate text.
    """
    
    license_plate_ = ''
    
    mapping = {0: dict_num_to_char, 
               1: dict_char_to_num, 
               2: dict_char_to_num, 
               3: dict_char_to_num, 
               4: dict_num_to_char,
               5: dict_num_to_char}
    
    for j in range(6):
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_

# func to check number in database
def check_plate_in_db(plate_number, database_ref):
    try:
        # Debugs
        # print(f"DEBUG: Checking plate_number (RAW): '{plate_number}'")
        # print(f"DEBUG: Length of plate_number: {len(plate_number)}")
        # print(f"DEBUG: ASCII codes: {[ord(c) for c in plate_number]}")
        ref_path = f'number_plates/{plate_number}'        
        # print(f"DEBUG: Full RTDB path: '{ref_path}'")

        ref = database_ref.reference(ref_path)
        data = ref.get()
        # return data # if need to check parking payment
        # Debug
        # print(f"DEBUG: Data returned from RTDB for '{ref_path}': {data}")

        if data is not None:
            return True # found number plate
        else:
            return False # number plate not found
        
    except Exception as e:
        print(f'Error accessing Firestore: {e}')
        return False
    

# func to check number in database
def check_plate_in_db(plate_number, database_ref):
    """
    Checks if a license plate number exists in the Firebase Realtime Database.

    Args:
        plate_number (str): The license plate number to search for.
        database_ref: A reference to the Firebase Realtime Database obtained from the Firebase SDK.

    Returns:
        bool: True if the license plate number is found in the database, False otherwise, or if an error occurs.

    Raises:
        Exception:  In case of an error accessing the database, the error is caught, printed to the console, and the function returns False.

    Details:
        The function uses the provided `database_ref` to access the database.
        It constructs the database path using the format `number_plates/{plate_number}`.
        Then, it attempts to retrieve data from this path.
        If data exists (is not None), the function returns True, indicating that the license plate number was found.
        If no data is found (is None), the function returns False.
        If any error occurs during database access (e.g., authorization issues, network problems), the function catches the exception, prints an error message, and returns False.
        The commented-out print statements are for debugging and can be uncommented to check input data, constructed paths, and returned data.
    """
    try:
        # Debugs
        # print(f"DEBUG: Checking plate_number (RAW): '{plate_number}'")
        # print(f"DEBUG: Length of plate_number: {len(plate_number)}")
        # print(f"DEBUG: ASCII codes: {[ord(c) for c in plate_number]}")
        ref_path = f'number_plates/{plate_number}'
        # print(f"DEBUG: Full RTDB path: '{ref_path}'")

        ref = database_ref.reference(ref_path)
        data = ref.get()
        # return data # 1 or 0 if need to check parking payment
        # Debug
        # print(f"DEBUG: Data returned from RTDB for '{ref_path}': {data}")

        if data is not None:
            return True # found number plate
        else:
            return False # number plate not found

    except Exception as e:
        print(f'Error accessing Firestore: {e}')
        return False