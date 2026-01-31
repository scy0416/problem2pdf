import streamlit as st
import numpy as np
import cv2
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from PIL import Image

def preprocess_image(uploaded_image):
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return img, processed

def seperate_image(img):
    h, w = img.shape[:2]

    top_bar = img[0:int(h*0.14), :]
    meta_row = img[int(h*0.14):int(h*0.18), 0:int(w*0.50)]
    body = img[int(h*0.19):int(h*0.78), :]
    bottom_bar = img[int(h*0.78):h, :]

    #prb_num = meta_row[:, 0:int(w*0.11)]
    #prb_answer = meta_row[:, int(w*0.18):int(w*0.30)]

    #return prb_num, prb_answer, body
    return meta_row, body

def crop_to_content_pixels(img_bgr, white_thresh=245, pad=20):
    h, w = img_bgr.shape[:2]
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray < white_thresh)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img_bgr

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    return img_bgr[y1:y2, x1:x2]

def extract_num_ans(img):
    h, w = img.shape[:2]

    prb_num = img[:, 0:int(w*0.35)]
    prb_ans = img[:, int(w*0.80):w]

    prb_num = crop_to_content_pixels(prb_num, white_thresh=245, pad=5)
    prb_ans = crop_to_content_pixels(prb_ans, white_thresh=245, pad=5)

    return prb_num, prb_ans

def concat_vertical_pad_right(images, bg_color=(255, 255, 255), gap=10):
    """
    images: OpenCV BGR 이미지 리스트
    bg_color: 배경색 (기본 흰색)
    gap: 이미지 사이 세로 간격(px)
    """
    # 1) None / 빈 이미지 제거
    imgs = [img for img in images if img is not None and img.size > 0]
    if not imgs:
        return None

    # 2) 기준 너비
    max_w = max(img.shape[1] for img in imgs)

    padded_imgs = []
    for img in imgs:
        h, w = img.shape[:2]

        if w < max_w:
            # 오른쪽에 흰색 패딩
            pad_w = max_w - w
            pad = np.full((h, pad_w, 3), bg_color, dtype=np.uint8)
            img = np.hstack([img, pad])

        padded_imgs.append(img)

    # 3) 전체 높이 계산
    total_h = sum(img.shape[0] for img in padded_imgs) + gap * (len(padded_imgs) - 1)

    # 4) 최종 캔버스
    canvas = np.full((total_h, max_w, 3), bg_color, dtype=np.uint8)

    y = 0
    for img in padded_imgs:
        h = img.shape[0]
        canvas[y:y+h, :] = img
        y += h + gap

    return canvas

def concat_horizontal(images, bg_color=(255, 255, 255), gap=10):
    imgs = [img for img in images if img is not None and img.size > 0]
    if not imgs:
        return None

    max_h = max(img.shape[0] for img in imgs)

    resized = []
    for img in imgs:
        h, w = img.shape[:2]
        if h != max_h:
            scale = max_h / h
            img = cv2.resize(img, (int(w * scale), max_h))
        resized.append(img)

    total_w = sum(img.shape[1] for img in resized) + gap * (len(resized) - 1)
    canvas = np.full((max_h, total_w, 3), bg_color, dtype=np.uint8)

    x = 0
    for img in resized:
        w = img.shape[1]
        canvas[:, x:x+w] = img
        x += w + gap

    return canvas

def start_pdf_in_memory(pagesize=A4):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=pagesize)
    return buffer, c

def add_image_to_pdf(
    c,
    pil_img: Image.Image,
    margin_mm=15,
    gap_mm=6,
    state=None,
):
    """
    state: {"y": current_y} 형태로 현재 커서 위치를 유지
    pil_img: PIL Image (RGB)
    """
    page_w, page_h = A4
    margin = margin_mm * mm
    gap = gap_mm * mm
    usable_w = page_w - 2 * margin

    if state is None:
        state = {"y": page_h - margin}

    y = state["y"]

    iw, ih = pil_img.size
    scale = usable_w / iw
    draw_w = usable_w
    draw_h = ih * scale

    # 공간 부족하면 새 페이지
    if y - draw_h < margin:
        c.showPage()
        y = page_h - margin

    x = margin
    y_draw = y - draw_h
    c.drawInlineImage(pil_img, x, y_draw, width=draw_w, height=draw_h)
    y = y_draw - gap

    state["y"] = y
    return state

def add_image_to_pdf_columns(
    c,
    pil_img: Image.Image,
    state: dict,
    n_cols: int = 2,
    margin_mm: float = 15,
    col_gap_mm: float = 6,
    row_gap_mm: float = 6,
):
    """
    - 페이지를 n_cols 컬럼으로 분할
    - 왼쪽 컬럼부터 위->아래로 누적
    - 공간 부족하면 다음 컬럼
    - 마지막 컬럼도 부족하면 새 페이지

    state 예시:
      {"page_w":..., "page_h":..., "col":0, "y":None}
    """
    page_w, page_h = A4
    margin = margin_mm * mm
    col_gap = col_gap_mm * mm
    row_gap = row_gap_mm * mm

    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    # 컬럼 너비 계산
    col_w = (usable_w - (n_cols - 1) * col_gap) / n_cols

    # state 초기화
    if not state:
        state = {"col": 0, "y": page_h - margin}
    if state.get("y") is None:
        state["y"] = page_h - margin
    if state.get("col") is None:
        state["col"] = 0

    col = state["col"]
    y = state["y"]

    iw, ih = pil_img.size

    # 기본: 컬럼 폭에 맞춰 비율 유지 스케일
    scale = col_w / iw
    draw_w = col_w
    draw_h = ih * scale

    # (안전장치) 너무 길어서 한 컬럼 높이를 초과하면 "높이에 맞춰 추가 축소"
    if draw_h > usable_h:
        scale2 = usable_h / draw_h
        draw_w *= scale2
        draw_h *= scale2

    # 현재 컬럼에서 그릴 x 계산 (왼쪽부터)
    def col_x(col_index: int) -> float:
        return margin + col_index * (col_w + col_gap)

    # 현재 컬럼에 공간이 있는지 확인
    if (y - draw_h) < margin:
        # 다음 컬럼으로
        col += 1
        y = page_h - margin

        # 마지막 컬럼도 넘어가면 새 페이지
        if col >= n_cols:
            c.showPage()
            col = 0
            y = page_h - margin

    x = col_x(col)
    y_draw = y - draw_h

    # 컬럼 폭(col_w)보다 더 줄어든 경우(높이 맞춤 축소) 왼쪽 정렬 유지
    c.drawInlineImage(pil_img, x, y_draw, width=draw_w, height=draw_h)

    # 다음 위치 갱신
    y = y_draw - row_gap

    state["col"] = col
    state["y"] = y
    return state

def finish_pdf_in_memory(buffer, c):
    c.save()
    buffer.seek(0)
    return buffer

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image")
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

# 제목
st.header("세현이만을 위한 기출문제 PDF 변환기")

# 경고
with st.container(border=True):
    st.subheader("사용법")
    st.write("아래의 이미지 입력을 통해서 **같은 회차**의 문제 스크린 샷을 **한 번에 전달**한다.")
    st.write("**유의사항**")
    st.write("- 되도록 같은 회차끼리만 전달할 것")
    st.write("- 스크린샷에는 정답이 표시되어 있어야 할 것")
    st.write("- 문제 위치를 옮기지 말 것(옮기면 문제생길 수 있음)")
    with st.expander("예시 스크린샷 이미지"):
        st.image("test_image.png")

# 이미지 전달 영역
uploaded_images = st.file_uploader("문제 전달", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
# for uploaded_image in uploaded_images:
#     st.markdown(f"### 문제 이미지")
#
#     original, processed = preprocess_image(uploaded_image)
#
#     meta_row, body = seperate_image(original)
#     meta_row = crop_to_content(meta_row, white_thresh=245, pad=5)
#     body = crop_to_content(body, white_thresh=245, pad=5)
#     prb_num, prb_ans = extract_num_ans(meta_row)
#     #st.image(prb_num)
#     #st.image(prb_ans)
#     #st.image(body)
#
#     header = concat_horizontal([prb_num, prb_ans], gap=20)
#     prb = concat_vertical_pad_right([header, body])
#     #st.image(prb)
n_cols = 3
if uploaded_images and st.button("PDF 생성"):
    buffer, c = start_pdf_in_memory()
    state = {}

    for f in uploaded_images:
        original, processed = preprocess_image(f)

        meta_row, body = seperate_image(original)
        meta_row = crop_to_content_pixels(meta_row, white_thresh=245, pad=5)
        body = crop_to_content_pixels(body, white_thresh=245, pad=5)
        prb_num, prb_ans = extract_num_ans(meta_row)

        header = concat_horizontal([prb_num, prb_ans], gap=20)
        prb = concat_vertical_pad_right([header, body])

        img_final = prb

        pil_img = cv2_to_pil(img_final)

        #state = add_image_to_pdf(c, pil_img, state=None if state["y"] is None else state)
        state = add_image_to_pdf_columns(
            c,
            pil_img,
            state=state,
            n_cols=n_cols,
            margin_mm=15,
            col_gap_mm=6,
            row_gap_mm=6
        )

    pdf_buffer = finish_pdf_in_memory(buffer, c)

    st.download_button(
        "PDF 다운로드",
        data=pdf_buffer,
        file_name="기출문제.pdf",
        mime="application/pdf"
    )