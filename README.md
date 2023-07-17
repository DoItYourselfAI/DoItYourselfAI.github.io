# diya blog

## Jekyll 테마

Skinny Bones

## devcontainer

- vscode 의 확장. 도커 컨테이너 내부에서 코드를 개발할 수 있게 해 준다.
- `.devcontainer` 디렉터리에 해당 설정들이 있다
- [참고 링크](https://code.visualstudio.com/docs/remote/containers)

## 멤버 추가하는 방법

- `_data/members.csv` 에 새로운 row 추가
- (optional) `images/profile` 디렉터리에 **500 x 500 크기 이하의 정사각형 프로필 사진을 올리고** `_data/members.csv` 에 프로필 파일 이름 추가

## What to do in Mac OS
준비물: [vscode](https://code.visualstudio.com/download) 와 [docker](https://docs.docker.com/docker-for-mac/install/)  

```bash
git clone [repository]
```  
  
vsc를 이용해 클론한 폴더를 연 다음,  
`Command + Shift + P`  
`>Remote-Containers: Reopen in Container` 선택  
  
Docker app이 실행 중이지 않으면 아래와 같은 에러가 나온다.  
```bash
Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?  
```  
당황하지 말고 앱을 설치해 실행해주자.  
[Docker](https://docs.docker.com/docker-for-mac/install/)
  
vsc에서 다시 `>Remote-Containers: Reopen in Container`를 선택하면 정상적으로 들어가진다.  
  
로컬에 띄우기
[참고링크](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll)  
  
vsc integrated terminal에서 새로운 터미널을 연 뒤,

```bash
bundle
bundle exec jekyll serve
```
  
Server address: 뒤의 주소를 Cmd+Click하면 디야 블로그가 떠있다.