```
cd ai-server/src

pyenv virtualenv 3.10.0 myenv
pyenv activate myenv

pip install -r requirements_1.txt
uvicorn main:app --reload
```

http://localhost:8080

```
curl --location --request GET 'http://localhost:8000/spam_filtering' \
--header 'Content-Type: application/json' \
--data '{
    "_id": "65509e7e35a5fe8ab15550a0",
  "domain": [
      "Phần mềm (Software)"
  ],
  "professional": [
      "Mọi người"
  ],
  "geographical": [
      "Vietnam"
  ],
  "ageRange": [
      0,
      29
  ],
  "outstand": [
      "Ứng dụng cao"
  ],
  "name": "\"Sàn kết nối ý tưởng với doanh nghiệp\"",
  "slogan": "\"Ý tưởng không còn là ý tưởng\"",
  "problem": "\"Những hệ thống đăng tải ý tưởng ngày nay chưa đáp ứng đủ nhu cầu của nhà sáng tạo. Các ý tưởng đăng lên nhưng không quá chú trọng về nội dung mà chỉ quan tâm đến số lượng.\"",
  "solution": "\"Hiện thực một hệ thống mới giúp người dùng có thể đăng tải những ý tưởng sáng kiến của bản thân về một vấn đề gì đó trong cuộc sống. Ở đây, các ý tưởng với độ hoàn thiện và được đánh giá cao sẽ có khả năng lớn trong việc kết hợp với doanh nghiệp trong ngành liên quan\"",
  "teamDescription": "\"<5 thành viên\"",
  "teamExperience": "\"Có kinh nghiệm trong việc xây dựng hệ thống. Có kiến thức về xử lý ngôn ngữ tự nhiên\"",
  "gender": "\"Male\"",
  "behavior": "\"Sàn ý tưởng giúp thu hẹp khoảng cách giữa ý tưởng và thực tế. Thông qua việc kết nối với doanh nghiệp, nhà sáng tạo có thể cơ hội, học hỏi thêm kinh nghiệm và những góp ý từ phía doanh nghiệp\"",
  "apps": "\"ytuongsangtao.net, ytuongsangtaohcm Quá nhiều ý tưởng, tuy nhiên nội dung ý tưởng không chất lượng\"",
  "currentDev": "\"[Hiện thực] Ý tưởng đang được hiện thực và thử nghiệm trong môi trường phát triển\"",
  "support": "undefined",
  "files": [
      "65509e7e35a5fe8ab15550a0/Cave-bg.png"
  ]
}'
```

```
curl --location --request GET 'http://localhost:8000/topk' \
--header 'Content-Type: application/json' \
--data '{
  "domain": [
      "Phần mềm (Software)"
  ],
  "problem": "\"Những hệ thống đăng tải ý tưởng ngày nay chưa đáp ứng đủ nhu cầu của nhà sáng tạo. Các ý tưởng đăng lên nhưng không quá chú trọng về nội dung mà chỉ quan tâm đến số lượng.\"",
  "acceptance_criteria": "Một hệ thống mới giúp người dùng có thể đăng tải những ý tưởng sáng kiến của bản thân về một vấn đề gì đó trong cuộc sống.",
  "constraints": "",
  "_id": "398723wsjb29"
}'
```
