# Installe winutils.exe + hadoop.dll (requis par Spark sous Windows) dans tools/hadoop/bin
# Usage : powershell -ExecutionPolicy Bypass -File scripts/setup_winutils.ps1
$ErrorActionPreference = "Stop"

$hadoopVersion = "hadoop-3.3.5"   # compatible avec le client Hadoop 3.3.x embarqué dans Spark 3.5
$baseUrl = "https://github.com/cdarlint/winutils/raw/master/$hadoopVersion/bin"
$binDir = Join-Path $PSScriptRoot "..\tools\hadoop\bin"

New-Item -ItemType Directory -Force -Path $binDir | Out-Null
foreach ($file in @("winutils.exe", "hadoop.dll")) {
    $target = Join-Path $binDir $file
    if (Test-Path $target) {
        Write-Host "[OK] $file déjà présent"
        continue
    }
    Write-Host "Téléchargement de $file ..."
    Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile $target -UseBasicParsing
    Write-Host "[OK] $file -> $target"
}
Write-Host "HADOOP_HOME sera détecté automatiquement par streaming/spark_streaming_redis.py"

