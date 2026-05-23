# Mispronunciation Detection and Diagnosis (MDD) — Problem Description

## 1. Tổng quan bài toán

Bài toán này là **MDD — Mispronunciation Detection and Diagnosis**, không phải ASR thông thường.

ASR thông thường tập trung vào việc chuyển đổi tín hiệu âm thanh thành văn bản:

```text
audio -> text
```

Trong khi đó, MDD tập trung vào việc đánh giá người nói có phát âm đúng theo cách phát âm chuẩn hay không.

Nói cách khác, hệ thống không chỉ cần biết speaker nói gì, mà cần biết speaker **có phát âm đúng với câu/từ/phoneme được yêu cầu hay không**.

---

## 2. Bản chất của MDD

MDD gồm hai nhiệm vụ chính:

### 2.1 Mispronunciation Detection

Detection trả lời câu hỏi:

```text
Speaker có phát âm sai không?
Sai ở vị trí nào?
```

Ví dụ:

```text
Canonical:  con trâu ăn cỏ
Transcript: con châu ăn cỏ
```

Ở đây speaker cần đọc từ `trâu`, nhưng thực tế phát âm thành `châu`.

Detection cần xác định:

```text
Từ/phoneme tương ứng với "trâu" bị phát âm sai.
```

### 2.2 Mispronunciation Diagnosis

Diagnosis trả lời câu hỏi:

```text
Sai như thế nào?
Phoneme chuẩn là gì?
Speaker đã phát âm thành phoneme gì?
Loại lỗi là substitution, deletion hay insertion?
```

Ví dụ ở mức phoneme:

```text
Canonical:  k on $ tr aw
Transcript: k on $ ch aw
```

Diagnosis cần xác định:

```text
tr -> ch
error_type = substitution
```

Do đó, bài toán này không dừng ở việc phát hiện lỗi, mà cần **chẩn đoán cụ thể lỗi phát âm ở mức phoneme**.

---

## 3. Input của bài toán

Input chính gồm:

```text
audio
+
canonical pronunciation / canonical text
```

Trong đó:

- `audio`: file âm thanh do speaker đọc.
- `canonical`: nội dung speaker được yêu cầu phát âm.
- `canonical pronunciation`: chuỗi phoneme chuẩn tương ứng với canonical text.

Ví dụ:

```text
audio: speaker đọc câu "con châu ăn cỏ"
canonical: "con trâu ăn cỏ"
canonical phones: k on $ tr aw $ ...
```

Mục tiêu là so sánh cách phát âm thực tế trong audio với cách phát âm chuẩn.

---

## 4. Dataset

The data used for the Mispronunciation Detection & Diagnosis Challenge is composed of two datasets. The first dataset [1] contains augmented recordings of adults speaking pairs of single-syllable Vietnamese words (released by MachinaX). The second dataset [2] features recordings of children aged 5 to 7, either speaking or reading Vietnamese sentences in passages or dialogues (released by SoICT-HUST and the Vietnam Psycho-Pedagogical Association).

Danh sách Pretrained Models được sử dụng như sau:
facebook/wav2vec2-base-100h, link ref: https://huggingface.co/facebook/wav2vec2-base-100h
nguyenvulebinh/wav2vec2-base-vietnamese-250h, link ref: https://huggingface.co/nguyenvulebinh/wav2vec2-base-vietnamese-250h
facebook/hubert-base-ls960, link ref: https://huggingface.co/facebook/hubert-base-ls960

### 4.1 `train.csv`

File `train.csv` gồm các trường chính:

| Cột          | Ý nghĩa                           |
| ------------ | --------------------------------- |
| `id`         | ID của sample                     |
| `path`       | Đường dẫn tới audio               |
| `canonical`  | Câu/từ speaker cần phát âm        |
| `transcript` | Câu/từ speaker thực tế đã phát âm |

Ví dụ:

| canonical      | transcript     |
| -------------- | -------------- |
| con trâu ăn cỏ | con châu ăn cỏ |

Ý nghĩa:

```text
Expected pronunciation: "trâu"
Actual pronunciation:   "châu"
```

Đây là dấu hiệu cho lỗi phát âm.

---

### 4.2 `train_phones.csv`

File `train_phones.csv` là phiên bản phoneme-level của dữ liệu.

Ví dụ:

| canonical    | transcript   |
| ------------ | ------------ |
| k on $ tr aw | k on $ ch aw |

Trong đó dấu `$` dùng để phân tách giữa các từ.

File này đặc biệt quan trọng vì MDD được đánh giá mạnh ở mức phoneme, thông qua các metric như PER và DER.

Vai trò chính của `train_phones.csv`:

- Tạo label phoneme-level.
- Align canonical phoneme với actual/spoken phoneme.
- Xác định lỗi substitution, deletion, insertion.
- Huấn luyện phoneme recognizer hoặc error detection head.
- Tính PER và hỗ trợ tính DER.

---

### 4.3 `lexicon_vmd.txt`

`lexicon_vmd.txt` là dictionary ánh xạ:

```text
word -> phoneme sequence
```

Ví dụ:

```text
trâu -> tr aw
châu -> ch aw
```

Vai trò của lexicon trong MDD rất quan trọng. Nó là cầu nối giữa dữ liệu word-level và phoneme-level:

```text
canonical text
-> lexicon / G2P
-> canonical phoneme sequence
```

và:

```text
transcript text
-> lexicon / G2P
-> spoken phoneme sequence
```

Nhờ đó, hệ thống có thể tạo nhãn huấn luyện ở mức phoneme.

---

## 5. Canonical vs Transcript

### 5.1 Canonical

`canonical` là nội dung speaker **được yêu cầu phát âm**.

```text
canonical = expected pronunciation
```

Ví dụ:

```text
trâu
```

### 5.2 Transcript

`transcript` là nội dung speaker **thực tế đã phát âm**.

```text
transcript = actual pronunciation
```

Ví dụ:

```text
châu
```

Do đó, cặp canonical/transcript cho biết lỗi phát âm:

```text
Expected: trâu
Actual:   châu
```

Ở mức phoneme:

```text
Expected: tr aw
Actual:   ch aw
```

Lỗi diagnosis:

```text
tr -> ch
```

---

## 6. Các loại lỗi trong MDD

MDD cần xử lý tối thiểu ba loại lỗi chính.

### 6.1 Substitution

Speaker phát âm một phoneme thành phoneme khác.

```text
canonical:  tr aw
spoken:     ch aw
```

Lỗi:

```text
tr -> ch
```

### 6.2 Deletion

Speaker bỏ sót một phoneme cần phát âm.

```text
canonical:  tr aw
spoken:        aw
```

Lỗi:

```text
tr -> <del>
```

### 6.3 Insertion

Speaker phát âm thêm một phoneme không có trong canonical.

```text
canonical:  aw
spoken:     ch aw
```

Lỗi:

```text
<ins> -> ch
```

Vì PER được tính từ substitution, deletion và insertion, internal representation của hệ thống cũng nên biểu diễn đủ ba loại lỗi này.

---

## 7. Alignment trong MDD

Alignment là bước trung tâm của bài toán MDD.

Trước khi detect hoặc diagnose lỗi, cần align hai chuỗi:

```text
canonical phoneme sequence
vs
spoken / predicted phoneme sequence
```

Ví dụ:

```text
canonical:  k on $ tr aw
spoken:     k on $ ch aw
```

Sau alignment:

```text
k   on   $   tr   aw
k   on   $   ch   aw
```

Diagnosis:

```text
tr -> ch
```

### 7.1 Alignment để tạo label training

Vì `train_phones.csv` đã có cả `canonical` và `transcript` ở mức phoneme, hướng hợp lý để tạo label là dùng **Levenshtein alignment**.

```text
canonical_phone
+
transcript_phone
-> Levenshtein alignment
-> phoneme-level labels
```

Label thu được gồm:

```text
correct
substitution
deletion
insertion
expected_phone
actual_phone
```

### 7.2 Alignment khi inference

Ở inference, thường không có transcript thực tế. Khi đó hệ thống cần dự đoán spoken phoneme từ audio:

```text
audio
-> phoneme recognizer / CTC decoder
-> predicted spoken phoneme
```

Sau đó align:

```text
canonical_phone
vs
predicted_spoken_phone
```

Từ alignment này suy ra lỗi phát âm.

---

## 8. Metrics

Score tổng có dạng:

```text
Score = 0.5 * F1_score + 0.4 * (1 - DER) + 0.1 * (1 - PER)
```

### 8.1 F1-score

F1-score đánh giá khả năng phát hiện đúng lỗi phát âm.

Nó tập trung vào câu hỏi:

```text
Model có detect đúng vị trí/phần bị phát âm sai không?
```

### 8.2 PER — Phoneme Error Rate

PER đo độ sai khác giữa chuỗi phoneme predicted và chuỗi phoneme reference.

Công thức:

```text
PER = (S + D + I) / N
```

Trong đó:

| Ký hiệu | Ý nghĩa                      |
| ------- | ---------------------------- |
| `S`     | số lỗi substitution          |
| `D`     | số lỗi deletion              |
| `I`     | số lỗi insertion             |
| `N`     | số phoneme trong chuỗi chuẩn |

PER càng thấp thì chuỗi phoneme predicted càng gần với chuỗi reference.

### 8.3 DER — Diagnosis Error Rate

DER là metric đánh giá phần diagnosis lỗi phát âm. Trong challenge này, công thức DER được định nghĩa trực tiếp trong `evaluate.py`.

Script tính DER như sau:

```text
DER = DE / total_actual_errors
```

Trong đó:

| Ký hiệu | Ý nghĩa |
| ------- | ------- |
| `DE` | Diagnosis Error: model phát hiện có lỗi nhưng chẩn đoán sai phoneme/loại lỗi |
| `total_actual_errors` | Tổng số lỗi phát âm thật trong ground truth |

Trong `evaluate.py`:

```text
DE = sub_sub1 + del_del1 + ins_ins1
total_actual_errors = TR + FA
```

Vì vậy:

- DER càng thấp càng tốt.
- Nếu model phát hiện đúng vị trí lỗi nhưng dự đoán sai phoneme lỗi, lỗi đó được tính vào `DE`.
- Nếu model bỏ sót lỗi thật, lỗi đó được tính vào `FA`.
- Nếu không có lỗi thật nào trong dữ liệu, DER được trả về `0.0`.

Ví dụ, nếu speaker phát âm:

```text
tr -> ch
```

thì model không chỉ cần phát hiện có lỗi, mà còn cần dự đoán đúng pair diagnosis:

```text
expected_phone = tr
actual_phone = ch
error_type = substitution
```

---

## 9. Format submission / prediction

Theo `evaluate.py`, file prediction cần là CSV có cột:

```text
predict
```

Ground truth cần có các cột:

```text
canonical,transcript
```

Hai file được so khớp theo thứ tự dòng, tức là dòng thứ `i` trong `results.csv` sẽ được so với dòng thứ `i` trong ground truth. Vì vậy thứ tự sample phải được giữ nguyên.

Trước khi align, `evaluate.py` xử lý chuỗi phoneme bằng:

```python
s.replace("*", "").replace("$", "").split()
```

Điều này có nghĩa là:

- Dấu `$` phân tách từ sẽ bị loại bỏ khi tính metric.
- Dấu `*` cũng bị loại bỏ.
- Metric thực sự chạy trên danh sách phoneme sau khi `split()`.

File `evaluate.py` in ra riêng:

```text
F1
PER
DER
```

Nó chưa trực tiếp tính score tổng:

```text
Score = 0.5 * F1_score + 0.4 * (1 - DER) + 0.1 * (1 - PER)
```
