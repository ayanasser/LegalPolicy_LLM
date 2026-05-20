# Closed-book smoke test — knowledge adapter


## Suite: `base_no_adapter`

**Aggregate**:
- verbatim (8 items): mean char-sim **0.07**, mean token-recall **0.03**, exact matches **0/8**
- reverse  (3 items): correct article-num **0/3**

### Verbatim recall (closed-book)

#### Article 1 (en) — char-sim 0.01, token-recall 0.07, exact=False
**Prompt:** Quote Article 1 of the Egyptian Civil Code exactly as it is written.

**Gold (384c):**
```
Provisions of laws govern all matters to which these provisions apply in letter or spirit. In the absence of a provision of a law that is applicable, the Judge will decide according to custom and in the absence of custom in accordance with the principles of Moslem Law. In the absence of such principles, the Judge will apply the principles of natural justice and the rules of equity.
```

**Prediction (375c, 7.6s):**
```
I'm sorry for any misunderstanding, but the Egyptian Civil Code is a very extensive document, and reproducing its entirety here would be impractical and potentially infringe on copyright. However, I can provide you with a specific article from the Egyptian Civil Code if you specify which article you are interested in. Please let me know which article you would like to see.
```

#### Article 280 (en) — char-sim 0.01, token-recall 0.00, exact=False
**Prompt:** Quote Article 280 of the Egyptian Civil Code exactly as it is written.

**Gold (283c):**
```
When there is solidarity between creditors, the debtor may pay the debt to anyone of them unless one of them objects to such payment. Solidarity does not prevent the debt being divided between the heirs of one of the joint and several creditors unless the debt itself is indivisible.
```

**Prediction (875c, 17.6s):**
```
I'm sorry for the misunderstanding, but I don't have direct access to the exact text of Article 280 of the Egyptian Civil Code as it is written in its original language. The Egyptian Civil Code is a complex document, and reproducing its exact wording in a different language, especially in a form that might not be legally binding, could lead to issues.

However, I can provide you with the English translation of Article 280 from the Egyptian Civil Code, which is as follows:

"Every marriage is valid if it is entered into by the parties in accordance with the provisions of this Code and the law."

This translation is based on the English version of the Egyptian Civil Code, which is widely used and recognized. If you need the original Arabic text, I would recommend consulting the official Egyptian Civil Code or a legal expert who can provide the exact Arabic wording.
```

#### Article 775 (en) — char-sim 0.16, token-recall 0.00, exact=False
**Prompt:** Quote Article 775 of the Egyptian Civil Code exactly as it is written.

**Gold (96c):**
```
Suretyship may be given without the knowledge and even in spite of the opposition of the debtor.
```

**Prediction (578c, 9.5s):**
```
I'm sorry for the misunderstanding, but I don't have direct access to the exact text of Article 775 of the Egyptian Civil Code. The Egyptian Civil Code is a complex legal document, and reproducing its exact text, especially a specific article, would require a thorough search of the official Egyptian Civil Code document.

However, I can provide you with a general overview or a translation of the article if you would like. If you need the exact wording, I would recommend consulting the official Egyptian Civil Code document or a legal expert who can provide the precise text.
```

#### Article 1068 (en) — char-sim 0.02, token-recall 0.10, exact=False
**Prompt:** Quote Article 1068 of the Egyptian Civil Code exactly as it is written.

**Gold (690c):**
```
The application shall be made by a summons to the third party holder and to the former owner, signed by the applicant or his representative holding a special mandate for this purpose. The applicant must deposit at the Caisse of the Court a sum which is sufficient to cover the cost of the sale by auction, but he shall have no right to a refund of expenses advanced by him if no higher price than that offered by the third party holder is obtained as a result of the auction. The failure to comply with any one of these conditions entails the nullity of the application. The applicant may not renounce his application without the consent of all the inscribed creditors and all the sureties.
```

**Prediction (861c, 14.2s):**
```
I'm sorry for the misunderstanding, but I don't have direct access to the exact text of Article 1068 of the Egyptian Civil Code as it is written in its original language. The Egyptian Civil Code is a complex document, and reproducing its exact wording in a different language, such as English, would require a thorough translation and verification process, which I don't have the capability to perform.

However, I can provide you with a general overview of what Article 1068 might cover, based on common topics in civil codes, such as property rights, inheritance, or family law. If you need a general description or explanation of what such an article might contain, I would be happy to assist with that. 

If you need the exact wording in English, you might want to consult the official Egyptian Civil Code or a legal expert who can provide the precise text.
```

#### Article 17 (ar) — char-sim 0.01, token-recall 0.06, exact=False
**Prompt:** اذكر نص المادة 17 من القانون المدني المصري حرفياً.

**Gold (275c):**
```
(١)يسري علي الميراث والوصية وسائر التصرفات المضافة إلي ما بعد الموت، قانون المورث أو الموصي أو من صدر منه التصرفات وقت موته. (٢)ومع ذلك يسري علي شكل الوصية، قانون الموصي وقت الإيصاء أو قانون البلد الذي تمت فيه الوصية وكذلك الحكم في شكل سائر التصرفات المضافة إلي ما بعد الموت.
```

**Prediction (1840c, 59.9s):**
```
عذرًا على الالتباس، ولكن كمساعد ذكي، ليس لدي القدرة على تقديم النص القانوني الأصلي حرفياً لأنني لا أملك الوصول إلى الإنترنت أو المكتبات القانونية. ومع ذلك، يمكنني أن أقدم لك نص المادة 17 من القانون المدني المصري بشكل عام، دون الاعتماد على مصدر خارجي:

"يعد العقد الذي يتعارض مع القانون أو العرف أو الأصل العام أو الأصل الخاص أو الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص بالعقد أو يتعارض مع الأصل العام أو الأصل الخاص بالعقد أو يتعارض مع الأصل الخاص
```

#### Article 836 (ar) — char-sim 0.02, token-recall 0.00, exact=False
**Prompt:** اذكر نص المادة 836 من القانون المدني المصري حرفياً.

**Gold (266c):**
```
(١)إذا أختلف الشركاء فى أقسام المال الشائع فعلى من يريد الخروج من الشيوع أن يكلف باقى الشركاء الحضور أمام المحكمة الجزئية . (٢)وتندب المحكمة إن رأت وجها لذلك خبيرا أو أكثر لتقويم المال الشائع وقسمته حصصا إن كان المال يقبل القسمة عينا دون أن يلحقه نقص كبير فى قسمته .
```

**Prediction (286c, 8.9s):**
```
عذرًا على الالتباس، ولكن المادة 836 من القانون المدني المصري ليست معلقًا عليها بشكل رسمي في الوقت الحالي. المادة 836 من القانون المدني المصري قد تختلف بناءً على التحديثات التي أجريت في القانون. يرجى تقديم مزيد من التفاصيل أو البحث في القانون المدني المصري الأصلي للحصول على النص الأصلي.
```

#### Article 990 (ar) — char-sim 0.08, token-recall 0.00, exact=False
**Prompt:** اذكر نص المادة 990 من القانون المدني المصري حرفياً.

**Gold (171c):**
```
(١)على المنتفع أن يبذل من العناية فى حفظ الشيء ما يبذله الشخص المعتاد . (٢)وهو مسئول عن هلاك الشيء ولو بسبب أجنبي إذا كان قد تأخر عن رده إلى صاحبة بعد انتهاء حق الانتفاع .
```

**Prediction (1717c, 49.0s):**
```
عذرًا على الالتباس، ولكن كمساعد ذكي، ليس لدي القدرة على الوصول إلى النص الفعلي للمادة 990 من القانون المدني المصري حرفياً، لأنني لا أملك الوصول إلى الإنترنت أو قاعدة بيانات قانونية محددة. ومع ذلك، يمكنني أن أقدم لك نص مقتضب للمادة 990 من القانون المدني المصري حسب المعلومات التي أملكها:

"يُعد العقد الذي يتعين على أحد الأطراف أن يعود إليه ما أودعه في حسابه البنكي أو في أي مكان آخر، عقدًا ملزِمًا، إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح، أو إذا كان العقد ينص على أن العقد يعود إلى الأطراف على الأرجح
```

#### Article 1112 (ar) — char-sim 0.26, token-recall 0.00, exact=False
**Prompt:** اذكر نص المادة 1112 من القانون المدني المصري حرفياً.

**Gold (186c):**
```
ينقضي حق الرهن الحيازى بانقضاء الدين المضمون ويعود معه إذا زال السبب الذى انقضى به الدين دون إخلال بالحقوق التى يكون الغير حسن النية قد كسبها قانونا فى الفترة ما بين انقضاء الحق وعودته .
```

**Prediction (274c, 7.4s):**
```
عذرًا على الالتباس، ولكن كمساعد ذكي، ليس لدي القدرة على الوصول إلى النص الفعلي للمادة 1112 من القانون المدني المصري حرفياً، لأنني لا أملك الوصول إلى الإنترنت أو الوصول إلى قوانين الدول الفعلية. ومع ذلك، يمكنني أن أقدم لك نص مقتضب لمثل هذه المادة إذا أردت ذلك. هل يمكنني ذلك؟
```

### Reverse lookup (closed-book)

#### ✗ expected=775 predicted=None (en, 0.4s)
**Prompt:** Which article of the Egyptian Civil Code contains the following provision? Reply with the article number only.

"Suretyship may be given without the knowledge and even in spite of the opposition of the d"

**Prediction:** `149`

#### ✗ expected=1 predicted=None (en, 0.3s)
**Prompt:** Which article of the Egyptian Civil Code contains the following provision? Reply with the article number only.

"Provisions of laws govern all matters to which these provisions apply in letter or spirit. In the absence of a provision of a law that is applicable, the Judge "

**Prediction:** `14`

#### ✗ expected=775 predicted=None (ar, 1.4s)
**Prompt:** أي مادة من القانون المدني المصري تتضمن النص التالي؟ اذكر رقم المادة فقط.

"تجوز كفالة المدين بغير عمله ، وتجوز أيضا رغم معارضته ."

**Prediction:** `رقم المادة هو 101 من قانون الأحوال الشخصية المصري.`

## Suite: `knowledge_adapter`

**Aggregate**:
- verbatim (8 items): mean char-sim **0.69**, mean token-recall **0.71**, exact matches **5/8**
- reverse  (3 items): correct article-num **1/3**

### Verbatim recall (closed-book)

#### Article 1 (en) — char-sim 0.21, token-recall 0.15, exact=False
**Prompt:** Quote Article 1 of the Egyptian Civil Code exactly as it is written.

**Gold (384c):**
```
Provisions of laws govern all matters to which these provisions apply in letter or spirit. In the absence of a provision of a law that is applicable, the Judge will decide according to custom and in the absence of custom in accordance with the principles of Moslem Law. In the absence of such principles, the Judge will apply the principles of natural justice and the rules of equity.
```

**Prediction (107c, 5.1s):**
```
The preceding provisions apply in the absence of a provision of the law or of an agreement to the contrary.
```

#### Article 280 (en) — char-sim 1.00, token-recall 1.00, exact=True
**Prompt:** Quote Article 280 of the Egyptian Civil Code exactly as it is written.

**Gold (283c):**
```
When there is solidarity between creditors, the debtor may pay the debt to anyone of them unless one of them objects to such payment. Solidarity does not prevent the debt being divided between the heirs of one of the joint and several creditors unless the debt itself is indivisible.
```

**Prediction (283c, 8.2s):**
```
When there is solidarity between creditors, the debtor may pay the debt to anyone of them unless one of them objects to such payment. Solidarity does not prevent the debt being divided between the heirs of one of the joint and several creditors unless the debt itself is indivisible.
```

#### Article 775 (en) — char-sim 0.25, token-recall 0.25, exact=False
**Prompt:** Quote Article 775 of the Egyptian Civil Code exactly as it is written.

**Gold (96c):**
```
Suretyship may be given without the knowledge and even in spite of the opposition of the debtor.
```

**Prediction (409c, 12.3s):**
```
Suretyship may be entered into in respect of a future debt, if the amount for which the guarantee is given is fixed beforehand. Suretyship may also be entered into in respect of a conditional liability. A surety however who has given his guarantee for a future debt, but has not fixed the duration of such guarantee, may revoke his guarantee at any time provided that the guaranteed debt has not been created.
```

#### Article 1068 (en) — char-sim 0.04, token-recall 0.25, exact=False
**Prompt:** Quote Article 1068 of the Egyptian Civil Code exactly as it is written.

**Gold (690c):**
```
The application shall be made by a summons to the third party holder and to the former owner, signed by the applicant or his representative holding a special mandate for this purpose. The applicant must deposit at the Caisse of the Court a sum which is sufficient to cover the cost of the sale by auction, but he shall have no right to a refund of expenses advanced by him if no higher price than that offered by the third party holder is obtained as a result of the auction. The failure to comply with any one of these conditions entails the nullity of the application. The applicant may not renounce his application without the consent of all the inscribed creditors and all the sureties.
```

**Prediction (471c, 12.5s):**
```
Every inscribed creditor and every surety of an inscribed debt has the right to apply for the sale of the property which the third party holder wishes to purge, provided that his application is made within thirty days of the date of the last formal summons. This period will be increased by the additional time allowed for distance between the actual and elected domicile of the creditor; this additional time allowed for distance shall not exceed thirty additional days.
```

#### Article 17 (ar) — char-sim 1.00, token-recall 1.00, exact=True
**Prompt:** اذكر نص المادة 17 من القانون المدني المصري حرفياً.

**Gold (275c):**
```
(١)يسري علي الميراث والوصية وسائر التصرفات المضافة إلي ما بعد الموت، قانون المورث أو الموصي أو من صدر منه التصرفات وقت موته. (٢)ومع ذلك يسري علي شكل الوصية، قانون الموصي وقت الإيصاء أو قانون البلد الذي تمت فيه الوصية وكذلك الحكم في شكل سائر التصرفات المضافة إلي ما بعد الموت.
```

**Prediction (275c, 14.7s):**
```
(١)يسري علي الميراث والوصية وسائر التصرفات المضافة إلي ما بعد الموت، قانون المورث أو الموصي أو من صدر منه التصرفات وقت موته. (٢)ومع ذلك يسري علي شكل الوصية، قانون الموصي وقت الإيصاء أو قانون البلد الذي تمت فيه الوصية وكذلك الحكم في شكل سائر التصرفات المضافة إلي ما بعد الموت.
```

#### Article 836 (ar) — char-sim 1.00, token-recall 1.00, exact=True
**Prompt:** اذكر نص المادة 836 من القانون المدني المصري حرفياً.

**Gold (266c):**
```
(١)إذا أختلف الشركاء فى أقسام المال الشائع فعلى من يريد الخروج من الشيوع أن يكلف باقى الشركاء الحضور أمام المحكمة الجزئية . (٢)وتندب المحكمة إن رأت وجها لذلك خبيرا أو أكثر لتقويم المال الشائع وقسمته حصصا إن كان المال يقبل القسمة عينا دون أن يلحقه نقص كبير فى قسمته .
```

**Prediction (266c, 22.8s):**
```
(١)إذا أختلف الشركاء فى أقسام المال الشائع فعلى من يريد الخروج من الشيوع أن يكلف باقى الشركاء الحضور أمام المحكمة الجزئية . (٢)وتندب المحكمة إن رأت وجها لذلك خبيرا أو أكثر لتقويم المال الشائع وقسمته حصصا إن كان المال يقبل القسمة عينا دون أن يلحقه نقص كبير فى قسمته .
```

#### Article 990 (ar) — char-sim 1.00, token-recall 1.00, exact=True
**Prompt:** اذكر نص المادة 990 من القانون المدني المصري حرفياً.

**Gold (171c):**
```
(١)على المنتفع أن يبذل من العناية فى حفظ الشيء ما يبذله الشخص المعتاد . (٢)وهو مسئول عن هلاك الشيء ولو بسبب أجنبي إذا كان قد تأخر عن رده إلى صاحبة بعد انتهاء حق الانتفاع .
```

**Prediction (171c, 13.2s):**
```
(١)على المنتفع أن يبذل من العناية فى حفظ الشيء ما يبذله الشخص المعتاد . (٢)وهو مسئول عن هلاك الشيء ولو بسبب أجنبي إذا كان قد تأخر عن رده إلى صاحبة بعد انتهاء حق الانتفاع .
```

#### Article 1112 (ar) — char-sim 1.00, token-recall 1.00, exact=True
**Prompt:** اذكر نص المادة 1112 من القانون المدني المصري حرفياً.

**Gold (186c):**
```
ينقضي حق الرهن الحيازى بانقضاء الدين المضمون ويعود معه إذا زال السبب الذى انقضى به الدين دون إخلال بالحقوق التى يكون الغير حسن النية قد كسبها قانونا فى الفترة ما بين انقضاء الحق وعودته .
```

**Prediction (186c, 9.6s):**
```
ينقضي حق الرهن الحيازى بانقضاء الدين المضمون ويعود معه إذا زال السبب الذى انقضى به الدين دون إخلال بالحقوق التى يكون الغير حسن النية قد كسبها قانونا فى الفترة ما بين انقضاء الحق وعودته .
```

### Reverse lookup (closed-book)

#### ✗ expected=775 predicted=770 (en, 1.8s)
**Prompt:** Which article of the Egyptian Civil Code contains the following provision? Reply with the article number only.

"Suretyship may be given without the knowledge and even in spite of the opposition of the d"

**Prediction:** `Article 770 of the Egyptian Civil Code.`

#### ✓ expected=1 predicted=1 (en, 1.5s)
**Prompt:** Which article of the Egyptian Civil Code contains the following provision? Reply with the article number only.

"Provisions of laws govern all matters to which these provisions apply in letter or spirit. In the absence of a provision of a law that is applicable, the Judge "

**Prediction:** `Article 1 of the Egyptian Civil Code.`

#### ✗ expected=775 predicted=780 (ar, 1.8s)
**Prompt:** أي مادة من القانون المدني المصري تتضمن النص التالي؟ اذكر رقم المادة فقط.

"تجوز كفالة المدين بغير عمله ، وتجوز أيضا رغم معارضته ."

**Prediction:** `المادة 780 من القانون المدني المصري.`