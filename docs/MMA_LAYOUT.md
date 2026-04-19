# MMA Register Layout Reference

## mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32

Computes `D = A × B + C` where:
- A: 16×64 INT4 matrix (row-major)
- B: 8×64 INT4 matrix (col-major, i.e., 64×8 stored as 8 rows)
- C/D: 16×8 INT32 matrix

### A-Operand (uint4 per thread)

```
Thread T (within warp of 32 threads):
  groupID = T / 4    → selects row pair (0-7)
  localID = T % 4    → selects k-column group

  a.x = A[groupID,     localID*8 .. localID*8+7]   (4 bytes = 8 INT4, k=0..31 portion)
  a.y = A[groupID+8,   localID*8 .. localID*8+7]   (same cols, rows 8-15)
  a.z = A[groupID,     32+localID*8 .. 32+localID*8+7]  (k=32..63 portion)
  a.w = A[groupID+8,   32+localID*8 .. 32+localID*8+7]
```

Visual (which bytes each thread's registers cover):

```
         k=0    k=8   k=16  k=24  k=32  k=40  k=48  k=56
row  0:  T0.x   T1.x  T2.x  T3.x  T0.z  T1.z  T2.z  T3.z
row  1:  T4.x   T5.x  T6.x  T7.x  T4.z  T5.z  T6.z  T7.z
  ...
row  7:  T28.x  T29.x T30.x T31.x T28.z T29.z T30.z T31.z
row  8:  T0.y   T1.y  T2.y  T3.y  T0.w  T1.w  T2.w  T3.w
  ...
row 15:  T28.y  T29.y T30.y T31.y T28.w T29.w T30.w T31.w
```

### B-Operand (uint2 per thread)

```
Thread T:
  groupID = T / 4    → selects which of 8 weight rows (= output column)
  localID = T % 4    → selects k-column group

  b.x = B[groupID, localID*8 .. localID*8+7]     (k=0..31)
  b.y = B[groupID, 32+localID*8 .. 32+localID*8+7]  (k=32..63)
```

### D-Output (4 × int32 per thread)

```
Thread T:
  row_lo = T / 4       (0-7)
  row_hi = row_lo + 8  (8-15)
  col0 = (T % 4) * 2
  col1 = col0 + 1

  c[0] = D[row_lo, col0]
  c[1] = D[row_lo, col1]
  c[2] = D[row_hi, col0]
  c[3] = D[row_hi, col1]
```

Visual output layout:

```
         col0  col1  col2  col3  col4  col5  col6  col7
row  0:  T0    T0    T1    T1    T2    T2    T3    T3
row  1:  T4    T4    T5    T5    T6    T6    T7    T7
  ...
row  7:  T28   T28   T29   T29   T30   T30   T31   T31
row  8:  T0    T0    T1    T1    T2    T2    T3    T3   (c[2],c[3])
  ...
row 15:  T28   T28   T29   T29   T30   T30   T31   T31
```

## ldmatrix.sync.aligned.x4.m8n8.shared.b16

Loads 4 × m8n8 tiles of 16-bit values from shared memory into registers.

### Input (shared memory)

16 rows × 16 bytes per row = 256 bytes per tile × 4 tiles

Thread T reads from: `smem[T % 16][(T / 16) * 4]` — this is the address each thread contributes.

Each thread reads one 128-bit (16-byte) row from shared memory. The hardware distributes the data across all threads in the warp according to the m8n8 layout.

### Output

`uint4 out` where each component holds data for one m8n8 tile.

### Mapping to INT4 MMA

For INT4 m16n8k64, a 16×32-byte region in shared memory maps to one A-fragment:
- Treat as 4 × m8n8 tiles of 16-bit values
- ldmatrix.x4 loads all 4 tiles
- Result is directly usable as MMA A-operand

For B, ldmatrix.x4 on a 16×32-byte region gives {x, y, z, w}. But we need to split this into two uint2 B-fragments for the two 8-column halves. The y/z swap rearranges so:
- `{x, y_new}` = first 8-column B-fragment
- `{z_new, w}` = second 8-column B-fragment

## Packed INT4 Format

Two signed 4-bit integers per byte:
```
byte = (high_nibble << 4) | (low_nibble & 0xF)

low_nibble  = byte & 0xF          → even element (sign-extended if ≥ 8)
high_nibble = (byte >> 4) & 0xF   → odd element  (sign-extended if ≥ 8)
```

Range: [-8, 7] for each 4-bit value.

Scale: `scale = max(|group|) / 7.0`
Dequantize: `value = quantized * scale`
