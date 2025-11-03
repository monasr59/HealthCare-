# stroke_gui_pdf_ar.py
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.filedialog import asksaveasfilename
import numpy as np
import pandas as pd
import joblib, os
from datetime import datetime

# PDF deps (Arabic support)
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Arabic shaping & bidi
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_OK = True
except Exception:
    ARABIC_OK = False

MODEL_PATH = "stroke_model.joblib"     # {'model': ..., 'features': [...]}
AR_FONT_FILE = "Amiri-Regular.ttf"  # ضع الخط بجوار السكربت (أو Amiri-Regular.ttf)
AR_FONT_NAME = "ArabicFont"            # اسم داخلي للتسجيل

class StrokeRiskGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stroke Risk – GUI")
        self.root.geometry("980x660")
        self.root.configure(bg="#f7f7f7")
        self.root.option_add("*Font", "Arial 10")

        # Load model & features
        self.model, self.features = None, None
        self._load_model()

        # Arabic UI label -> EXACT trained feature name
        self.ar2feat = {
            "ألم في الصدر": "Chest Pain",
            "ضيق في التنفس": "Shortness of Breath",
            "اضطراب نبضات القلب": "Irregular Heartbeat",
            "التعب و ضعط مرتفع": "Fatigue & Weakness",
            "دوخة": "Dizziness",
            "تورم (وذمة)": "Swelling (Edema)",
            "ألم في الرقبة / العد / الكتف / الظهر": "Pain in Neck/Jaw/Shoulder/Back",
            "التعرق الزائد": "Excessive Sweating",
            "السعال المستمر": "Persistent Cough",
            "الغثيان/القيء": "Nausea/Vomiting",
            "ضعط دم مرتفع": "High Blood Pressure",
            "أرتجاع في الصدر (النساط)": "Chest Discomfort (Activity)",
            "الأيدي الباردة / القدمين": "Cold Hands/Feet",
            "الشخير/توقف التنفس أثناء النوم": "Snoring/Sleep Apnea",
            "الغلق / الشعور بالهلاك": "Anxiety/Feeling of Doom",
            # "Age" عمود العمر
        }

        # سجل الخط العربي للـPDF (لو متوفر)
        self._register_arabic_font()

        self._build_ui()

    # ---------- Model ----------
    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("خطأ", f"الملف غير موجود: {MODEL_PATH}\nاحفظ الموديل من النوتبوك أولًا.")
            return
        data = joblib.load(MODEL_PATH)
        if not (isinstance(data, dict) and "model" in data and "features" in data):
            messagebox.showerror("خطأ", "تأكد أن الملف يحتوي {'model','features'}.")
            return
        self.model = data["model"]
        self.features = data["features"]
        print(f"Loaded model with {len(self.features)} features.")

    # ---------- UI ----------
    def _build_ui(self):
        main = tk.Frame(self.root, bg="#f7f7f7")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Top info
        info = tk.Frame(main, bg="#f7f7f7")
        info.pack(fill=tk.X, pady=(0, 10))
        tk.Label(info, text="الاسم", bg="#f7f7f7").pack(side=tk.RIGHT, padx=6)
        self.name_entry = tk.Entry(info, width=28); self.name_entry.pack(side=tk.RIGHT, padx=6)

        tk.Label(info, text="العمر", bg="#f7f7f7").pack(side=tk.RIGHT, padx=6)
        self.age_spin = tk.Spinbox(info, from_=1, to=120, width=8); self.age_spin.pack(side=tk.RIGHT, padx=6)

        tk.Label(info, text="الجنس", bg="#f7f7f7").pack(side=tk.RIGHT, padx=6)
        self.gender_combo = ttk.Combobox(info, width=12, state="readonly", values=("ذكر", "أنثى"))
        self.gender_combo.current(0); self.gender_combo.pack(side=tk.RIGHT, padx=6)

        # Symptoms
        tk.Label(main, text="الأعراض", font=("Arial", 14, "bold"), bg="#f7f7f7").pack(pady=(4, 8))
        cont = tk.Frame(main, bg="#f7f7f7"); cont.pack(fill=tk.BOTH, expand=True)
        left = tk.Frame(cont, bg="#f7f7f7"); right = tk.Frame(cont, bg="#f7f7f7")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.syms_left, self.syms_right = {}, {}
        left_syms = [
            "التعب و ضعط مرتفع", "ألم في الصدر", "ضيق في التنفس",
            "اضطراب نبضات القلب", "دوخة", "تورم (وذمة)",
            "ألم في الرقبة / العد / الكتف / الظهر",
        ]
        right_syms = [
            "التعرق الزائد", "السعال المستمر", "الغثيان/القيء",
            "ضعط دم مرتفع", "أرتجاع في الصدر (النساط)",
            "الأيدي الباردة / القدمين", "الشخير/توقف التنفس أثناء النوم",
            "الغلق / الشعور بالهلاك",
        ]
        for s in left_syms:  self._sym_row(left, s, self.syms_left)
        for s in right_syms: self._sym_row(right, s, self.syms_right)

        # Threshold
        thrf = tk.Frame(main, bg="#f7f7f7"); thrf.pack(fill=tk.X, pady=(8, 0))
        tk.Label(thrf, text="عتبة القرار (0..1)", bg="#f7f7f7").pack(side=tk.RIGHT, padx=6)
        self.thr_var = tk.DoubleVar(value=0.5)
        tk.Entry(thrf, width=6, textvariable=self.thr_var).pack(side=tk.RIGHT)

        # Buttons
        btns = tk.Frame(main, bg="#f7f7f7"); btns.pack(pady=(12, 0))
        tk.Button(btns, text="توقّع", bg="#e0e0e0", padx=24, pady=6,
                  command=self.predict_minimal).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="تصدير PDF", bg="#DCDCDC", padx=18, pady=6,
                  command=self.export_pdf).pack(side=tk.LEFT, padx=6)

        # Output (centered)
        out = tk.Frame(main, bg="#f7f7f7")
        out.pack(fill=tk.BOTH, pady=(20, 0), expand=True)
        tk.Label(out, text="النتيجة المختصرة", bg="#f7f7f7", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        self.short_out = tk.Label(out, text="—", bg="#f7f7f7", font=("Arial", 16, "bold"), fg="#333")
        self.short_out.pack(expand=True)

    def _sym_row(self, parent, ar_label, store):
        row = tk.Frame(parent, bg="#f7f7f7"); row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=ar_label, bg="#f7f7f7", anchor="e").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=6)
        v = tk.StringVar(value="")
        store[ar_label] = v
        cb = ttk.Combobox(row, textvariable=v, values=["", "Yes", "No"], state="readonly", width=12)
        cb.pack(side=tk.LEFT, padx=6)

    # ---------- Arabic helpers ----------
    def _register_arabic_font(self):
        if REPORTLAB_OK and os.path.exists(AR_FONT_FILE):
            try:
                pdfmetrics.registerFont(TTFont(AR_FONT_NAME, AR_FONT_FILE))
                print(f"Arabic font registered: {AR_FONT_FILE}")
            except Exception as e:
                print("Font register error:", e)

    def rtl(self, text):
        """Shape + bidi for Arabic string; safe for empty/None."""
        if not text:
            return ""
        if not ARABIC_OK:
            # fallback: return as-is (قد يظهر متقطع)
            return text
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)

    # ---------- Predict helpers ----------
    def _get_symptoms_booleans(self):
        syms = {}
        for k, v in self.syms_left.items():  syms[k] = (v.get() == "Yes")
        for k, v in self.syms_right.items(): syms[k] = (v.get() == "Yes")
        return syms

    def _get_pos_index(self):
        if hasattr(self.model, "classes_"):
            classes = list(self.model.classes_)
            return classes.index(1) if 1 in classes else (len(classes) - 1)
        return 1

    def _build_row(self):
        if not self.model or not self.features:
            raise RuntimeError("Model/features not loaded.")
        syms = self._get_symptoms_booleans()
        row = pd.DataFrame(np.zeros((1, len(self.features))), columns=self.features)

        unmapped = []
        for ar, is_yes in syms.items():
            feat = self.ar2feat.get(ar)
            if feat in row.columns:
                row.loc[0, feat] = 1 if is_yes else 0
            else:
                unmapped.append((ar, feat))

        if "Age" in row.columns:
            try:
                row.loc[0, "Age"] = float(self.age_spin.get())
            except Exception:
                row.loc[0, "Age"] = 0.0

        # Debug print
        print("---- DEBUG: built input row ----")
        print("Non-zero columns:", [c for c in row.columns if float(row.loc[0, c]) != 0.0])
        if unmapped:
            print("UNMAPPED labels (fix ar2feat):", unmapped)
        print(row.to_string(index=False))
        print("--------------------------------")
        return row, syms

    # ---------- Predict (short) ----------
    def predict_minimal(self):
        try:
            if not self.model:
                messagebox.showwarning("تنبيه", "الموديل غير محمّل.")
                return
            X_new, _ = self._build_row()
            thr = float(self.thr_var.get() or 0.5)
            thr = min(max(thr, 0.0), 1.0)

            if hasattr(self.model, "predict_proba"):
                pos_idx = self._get_pos_index()
                p1 = float(self.model.predict_proba(X_new)[0, pos_idx])
                y  = int(p1 >= thr)
                pct = p1 * 100.0
            else:
                y  = int(self.model.predict(X_new)[0]); pct = 100.0 if y == 1 else 0.0

            level = "عالية" if y == 1 else "منخفضة"
            self.short_out.config(text=f"الحالة: {level} — النسبة: {pct:.2f}% (عتبة {thr:.2f})")
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل التوقّع:\n{e}")

    # ---------- Build report dict ----------
    def build_report_data(self):
        syms = {}
        for k, v in self.syms_left.items():  syms[k] = ("Yes" if v.get() == "Yes" else "No")
        for k, v in self.syms_right.items(): syms[k] = ("Yes" if v.get() == "Yes" else "No")

        X_new, _ = self._build_row()
        thr = float(self.thr_var.get() or 0.5)
        thr = min(max(thr, 0.0), 1.0)

        if hasattr(self.model, "predict_proba"):
            pos_idx = self._get_pos_index()
            p1 = float(self.model.predict_proba(X_new)[0, pos_idx])
            y  = int(p1 >= thr); pct = p1 * 100.0
        else:
            y  = int(self.model.predict(X_new)[0]); pct = 100.0 if y == 1 else 0.0

        level = "عالية" if y == 1 else "منخفضة"
        return {
            "name": self.name_entry.get().strip(),
            "age": self.age_spin.get(),
            "gender": self.gender_combo.get(),
            "threshold": thr,
            "level": level,
            "percent": pct,
            "symptoms": syms,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    # ---------- Export Arabic PDF ----------
    def export_pdf(self):
        try:
            if not self.model:
                messagebox.showwarning("تنبيه", "الموديل غير محمّل.")
                return
            if not REPORTLAB_OK:
                messagebox.showerror("خطأ", "ReportLab غير منصّب.\nنفّذ: pip install reportlab")
                return
            if not os.path.exists(AR_FONT_FILE):
                messagebox.showerror("خطأ", f"لم يتم العثور على ملف الخط:\n{AR_FONT_FILE}\nضعه بجوار السكربت.")
                return

            data = self.build_report_data()

            default_name = f"تقرير_السكتة_{data['name'] or 'مريض'}_{data['created_at'].replace(':','-').replace(' ','_')}.pdf"
            filepath = asksaveasfilename(defaultextension=".pdf",
                                         filetypes=[("PDF", "*.pdf")],
                                         initialfile=default_name,
                                         title="حفظ التقرير كـ PDF")
            if not filepath:
                return

            c = canvas.Canvas(filepath, pagesize=A4)
            W, H = A4
            x_left  = 20 * mm    # الهامش الأيسر (هنرسم يمين لسطر RTL)
            x_right = W - 20 * mm
            y = H - 25 * mm

            def draw_rtl(txt, x, y, size=12, bold=False):
                c.setFont(AR_FONT_NAME, size)
                shaped = self.rtl(txt)
                # نكتب يمين لليسار بمحاذاة يمين
                c.drawRightString(x, y, shaped)

            # عنوان
            draw_rtl("تقرير تقييم خطر السكتة الدماغية", x_right, y, size=18); y -= 12 * mm

            # بيانات المريض
            draw_rtl(f"الاسم: {data['name'] or '-'}", x_right, y, size=12); y -= 7 * mm
            draw_rtl(f"العمر: {data['age'] or '-'}", x_right, y, size=12); y -= 7 * mm
            draw_rtl(f"النوع: {data['gender'] or '-'}", x_right, y, size=12); y -= 10 * mm

            # النتيجة
            draw_rtl(f"الحالة: {data['level']}   —   النسبة: {data['percent']:.2f}%   (العتبة: {data['threshold']:.2f})",
                     x_right, y, size=13); y -= 12 * mm

            # الأعراض
            draw_rtl("الأعراض:", x_right, y, size=13); y -= 8 * mm

            # عمودان RTL
            items = list(data["symptoms"].items())
            half = (len(items) + 1)//2
            right_items = items[:half]   # العمود الأيمن
            left_items  = items[half:]   # العمود الأيسر

            line_h = 7 * mm
            y_right_col = y
            y_left_col  = y
            for k, v in right_items:
                txt = f"- {k}: {'نعم' if v=='Yes' else 'لا'}"
                draw_rtl(txt, x_right, y_right_col, size=12)
                y_right_col -= line_h
                if y_right_col < 25*mm:
                    c.showPage(); y_right_col = H - 25*mm

            # العمود الثاني نحدّد له x أبعد لليسار
            x_col2 = x_right - (W - 40*mm) / 2
            for k, v in left_items:
                txt = f"- {k}: {'نعم' if v=='Yes' else 'لا'}"
                draw_rtl(txt, x_col2, y_left_col, size=12)
                y_left_col -= line_h
                if y_left_col < 25*mm:
                    c.showPage(); y_left_col = H - 25*mm

            # تذييل
            y_footer = 15 * mm
            draw_rtl(f"تاريخ إنشاء التقرير: {data['created_at']}", x_right, y_footer, size=10)

            c.save()
            messagebox.showinfo("تم الحفظ", f"تم حفظ التقرير:\n{filepath}")
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل إنشاء PDF:\n{e}")

def main():
    root = tk.Tk()
    app = StrokeRiskGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
