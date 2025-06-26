+++
date = '2025-06-16T18:09:01+08:00'
title = 'Dapper'
+++

## Dapper

- Dapper 的作用

创建一致的构建环境：Dapper 可以将任何现有的构建工具包装在一个一致的环境中。这意味着无论开发者使用何种操作系统、开发环境或工具链，都可以在相同的环境下进行软件的构建和修改，避免了因环境差异导致的构建失败或结果不一致的问题。
简化构建过程：它允许人们从源代码构建你的软件或对其进行修改，而无需担心设置构建环境。开发者无需手动安装和配置各种依赖的构建工具和库，大大简化了软件的构建和开发流程。

- Dapper 的工作原理

基于 Dockerfile 的构建：Dapper 的工作方式基于 Dockerfile，这是一种用于定义 Docker 镜像构建过程的文件。你需要在你的代码仓库的根目录下创建一个名为 Dockerfile.dapper 的文件，该文件中定义了构建环境所需的依赖、工具和配置。
构建镜像并执行容器：Dapper 会根据 Dockerfile.dapper 构建一个 Docker 镜像，然后基于这个镜像启动一个容器。在这个容器中，Dapper 会执行构建操作，确保构建过程在一个隔离且一致的环境中进行。
源文件的处理：Dapper 会将源代码文件复制到容器中指定的位置，以便在容器内进行构建。构建完成后，它会将生成的构建产物（如可执行文件、库文件等）复制回宿主机的指定位置，或者根据你的选择使用绑定挂载（bind mounting）的方式，直接在宿主机和容器之间共享文件，这样可以更方便地访问和使用构建产物。

- [GitHub](https://github.com/rancher/dapper)

dapper ENV:

- DAPPER_SOURCE: 挂载或者复制项目目录(.)到容器中的位置 (docker run -v .:${DAPPER_SOURCE} build-image)
- DAPPER_CP: 挂载或者复制本地文件(DAPPER_CP)到容器中的位置，位置由DAPPER_SOURCE指定 (docker run -v ${DAPPER_CP}:/source/ build-image)
- DAPPER_OUTPUT: 将构建容器中的文件复制到本地的位置 (docker cp ${DAPPER_SOURCE}/${DAPPER_OUTPUT} .)
- DAPPER_DOCKER_SOCKET: 挂载docker.sock到容器
- DAPPER_RUN_ARGS: docker run 参数
- DAPPER_ENV:  docker run 环境变量

示例：

```dockerfile
FROM golang:1.15
RUN go get github.com/tools/godep
ENV DAPPER_SOURCE /go/src/github.com/rancher/dapper
ENV DAPPER_OUTPUT bin
WORKDIR ${DAPPER_SOURCE}
ENTRYPOINT ["./script/build"]
```
